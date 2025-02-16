import os
import gc
import random
import argparse
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
import numpy as np

from datasets import load_dataset

from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor, resize

import ImageReward as RM

from utils.loading import load_models
from utils import p2p, generation
from diffusers import DDPMScheduler

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iCD-SD on MS-COCO")
    parser.add_argument(
        "--model_id",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="Hugging Face Model ID (Stable Diffusion v1.5 base).",
    )
    parser.add_argument(
        "--forward_checkpoint",
        type=str,
        help="Path to the forward consistency model .safetensors or .pt",
        required=True,
    )
    parser.add_argument(
        "--reverse_checkpoint",
        type=str,
        help="Path to the reverse consistency model .safetensors or .pt",
        required=True,
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Precision for the loaded models and tensors.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for both generation and metrics.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="How many images from MS-COCO to evaluate. (max ~25K or 40K+ for test sets)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale for iCD re-generation.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Tau parameter for dynamic guidance (if < 1.0).",
    )
    parser.add_argument(
        "--num_ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps (same as in iCD code).",
    )
    parser.add_argument(
        "--reverse_timesteps",
        nargs="+",
        type=int,
        default=[259, 519, 779, 999],
        help="Timesteps for the reverse consistency pass (must match your training).",
    )
    parser.add_argument(
        "--forward_timesteps",
        nargs="+",
        type=int,
        default=[19, 259, 519, 779],
        help="Timesteps for the forward consistency pass (must match your training).",
    )
    parser.add_argument(
        "--start_timestep",
        type=int,
        default=19,
        help="At which DDIM step iCD starts reconstruction (19 by default).",
    )
    parser.add_argument(
        "--max_forward_timestep_index",
        type=int,
        default=49,
        help="Typically num_ddim_steps-1. Tells iCD solver when to stop forward steps.",
    )
    parser.add_argument(
        "--image_reward_model",
        type=str,
        default="ImageReward-v1.0",
        help="Which ImageReward model to load. If empty, skip ImageReward metric.",
    )
    return parser.parse_args()


@torch.no_grad()
def encode_vae_images(ldm_stable, images: torch.Tensor) -> torch.Tensor:
    moments = ldm_stable.vae.encode((images * 2.0 - 1.0)).latent_dist
    return moments.sample() * 0.18215


@torch.no_grad()
def decode_vae_latents(ldm_stable, latents: torch.Tensor) -> torch.Tensor:
    scaled_latents = latents / 0.18215
    imgs = ldm_stable.vae.decode(scaled_latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def compute_clip_score_all(gen_images, prompts, device="cuda", batch_size=32):
    n = len(prompts)
    scores = []

    if gen_images.max() <= 1.0:
        gen_images = (gen_images * 255.0).clamp(0, 255).to(torch.uint8)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = gen_images[start:end].to(device, non_blocking=True)
        batch_prompts = prompts[start:end]
        with torch.inference_mode():
            batch_score = clip_score(
                batch_imgs,
                batch_prompts,
                model_name_or_path="openai/clip-vit-base-patch16",
            )
        scores.append(batch_score.mean().item())

    return float(sum(scores) / len(scores))


def compute_fid_torchmetrics(real_images, gen_images, device="cuda", batch_size=32):
    fid_metric = FrechetInceptionDistance().to(device)
    n = real_images.shape[0]

    for idx in range(0, n, batch_size):
        end = min(idx + batch_size, n)
        real_batch = real_images[idx:end]
        gen_batch = gen_images[idx:end]

        if real_batch.max() <= 1.0:
            real_batch = (real_batch * 255.0).clamp(0, 255)
        if gen_batch.max() <= 1.0:
            gen_batch = (gen_batch * 255.0).clamp(0, 255)

        real_batch = real_batch.to(torch.uint8).to(device)
        gen_batch = gen_batch.to(torch.uint8).to(device)

        fid_metric.update(real_batch, real=True)
        fid_metric.update(gen_batch, real=False)

    return fid_metric.compute().item()


def preprocess_image(pil_img: Image.Image, size=(512, 512), dtype=torch.float16):
    pil_img = pil_img.convert("RGB")
    pil_img = resize(to_tensor(pil_img), size)
    if dtype == torch.float16:
        pil_img = pil_img.half()
    return pil_img


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\nLoading iCD + SD1.5 models ...")

    teacher_ckpt = None
    ldm_stable, reverse_cons_model, forward_cons_model = load_models(
        model_id=args.model_id,
        device=device,
        forward_checkpoint=args.forward_checkpoint,
        reverse_checkpoint=args.reverse_checkpoint,
        r=64,
        w_embed_dim=512,
        teacher_checkpoint=teacher_ckpt,
    )

    tokenizer = ldm_stable.tokenizer

    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    solver = generation.Generator(
        model=ldm_stable,
        noise_scheduler=noise_scheduler,
        n_steps=args.num_ddim_steps,
        forward_cons_model=forward_cons_model,
        forward_timesteps=args.forward_timesteps,
        reverse_cons_model=reverse_cons_model,
        reverse_timesteps=args.reverse_timesteps,
        num_endpoints=len(args.reverse_timesteps),
        num_forward_endpoints=len(args.forward_timesteps),
        max_forward_timestep_index=args.max_forward_timestep_index,
        start_timestep=args.start_timestep,
    )

    p2p.NUM_DDIM_STEPS = args.num_ddim_steps
    p2p.tokenizer = tokenizer
    p2p.device = device

    rm_model = None
    if args.image_reward_model:
        try:
            rm_model = RM.load(args.image_reward_model)
        except:
            print(f"Could not load ImageReward model {args.image_reward_model}")
            rm_model = None

    print("\nLoading MS-COCO test set (bitmind/MS-COCO)...")
    data_files = {
        "test": "data/test-*-of-*.parquet",
    }
    dataset = load_dataset(
        "bitmind/MS-COCO",
        data_files=data_files,
        split="test",
        verification_mode="no_checks",
    )
    print("Total samples in test split:", len(dataset))

    sample_size = min(args.sample_size, len(dataset))
    subset_indices = random.sample(range(len(dataset)), sample_size)
    dataset = dataset.select(subset_indices)
    print(f"Using random subset of size: {len(dataset)}")

    real_images_list = []
    rec_images_list = []
    prompts_list = []

    latent_mse_vals = []
    rec_mse_vals = []
    image_reward_vals = []

    global_generator = torch.Generator(device=device).manual_seed(1337)

    controller = p2p.AttentionStore()

    print("\nStarting evaluation loop...")
    for start_i in tqdm(
        range(0, len(dataset), args.batch_size), desc="Evaluation Loop"
    ):
        end_i = min(start_i + args.batch_size, len(dataset))
        batch = dataset[start_i:end_i]

        pil_images = []
        prompts = []
        for i in range(end_i - start_i):
            img = batch["image"][i].convert("RGB")
            pil_images.append(img)
            prompts.append(batch["sentences"][i]["raw"])

        real_imgs = []
        for img in pil_images:
            real_imgs.append(preprocess_image(img, size=(512, 512), dtype=dtype))
        real_imgs = torch.stack(real_imgs, dim=0).to(device)

        with torch.no_grad():
            latents_init = encode_vae_images(ldm_stable, real_imgs)

        with torch.no_grad():
            rec_images_icd, _ = generation.runner(
                guidance_scale=args.guidance_scale,
                tau1=args.tau,  # tau param if you want dynamic guidance < 1.0
                tau2=args.tau,  # typically set the same as tau1
                is_cons_forward=True,
                model=reverse_cons_model,
                w_embed_dim=512,
                solver=solver,
                prompt=prompts,
                controller=controller,
                generator=global_generator,
                latent=latents_init,
                return_type="pt",
            )
            # rec_images_icd.shape => (B, 3, 512, 512) in [0,1]

        mse_img = F.mse_loss(rec_images_icd, real_imgs).item()
        rec_mse_vals.append(mse_img)

        # For demonstration, we won't compute "latent_mse_vals" unless you adapt
        # the iCD code to return the final latents. We'll just store zero for now:
        latent_mse_vals.append(0.0)

        if rm_model is not None:
            batch_ir_scores = []
            rec_images_pil = []
            for b in range(rec_images_icd.size(0)):
                arr = rec_images_icd[b].permute(1, 2, 0).cpu().float().numpy()
                arr = np.clip(arr, 0, 1) * 255
                arr = arr.astype(np.uint8)
                rec_images_pil.append(Image.fromarray(arr))

            for prompt_i, pil_img in zip(prompts, rec_images_pil):
                score_i = rm_model.score(prompt_i, pil_img)
                batch_ir_scores.append(score_i)
            image_reward_vals.append(float(np.mean(batch_ir_scores)))
        else:
            image_reward_vals.append(0.0)

        real_images_list.append(real_imgs.cpu())
        rec_images_list.append(rec_images_icd.cpu())
        prompts_list.extend(prompts)

        del real_imgs, rec_images_icd, latents_init
        gc.collect()
        torch.cuda.empty_cache()

    print("\n==== Evaluation Metrics ====")
    real_images_all = torch.cat(real_images_list, dim=0)
    rec_images_all = torch.cat(rec_images_list, dim=0)

    print("Computing FID...")
    fid_value = compute_fid_torchmetrics(
        real_images_all, rec_images_all, device=device, batch_size=args.batch_size
    )
    print(f"FID Score: {fid_value:.4f}")

    print("Computing CLIP Score...")
    clip_value = compute_clip_score_all(
        rec_images_all, prompts_list, device=device, batch_size=args.batch_size
    )
    print(f"CLIP Score: {clip_value:.4f}")

    mean_latent_mse = float(np.mean(latent_mse_vals))
    print(f"Mean Latent MSE (placeholder): {mean_latent_mse:.6f}")

    mean_rec_mse = float(np.mean(rec_mse_vals))
    print(f"Mean Reconstruction MSE (pixel space): {mean_rec_mse:.6f}")

    mean_ir_score = float(np.mean(image_reward_vals))
    if rm_model is not None:
        print(f"Mean ImageReward Score: {mean_ir_score:.4f}")
    else:
        print("ImageReward was not computed (no model loaded).")


if __name__ == "__main__":
    main()
