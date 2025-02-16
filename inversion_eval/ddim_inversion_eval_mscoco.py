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

# Hugging Face datasets
from datasets import load_dataset

# Diffusers
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0

# ImageReward
import ImageReward as RM

# Torchmetrics
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor, resize

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDv1.5 Inversion on MS-COCO")
    parser.add_argument(
        "--model_id",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="Hugging Face Model ID for stable diffusion v1.5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation and metrics",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for the pipeline and stored tensors",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for both encoding/decoding and metrics",
    )
    parser.add_argument(
        "--num_inversion_steps",
        type=int,
        default=50,
        help="Number of steps for latents inversion",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=25010,
        help="How many images from MS-COCO to evaluate",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale for inversion",
    )
    return parser.parse_args()


@torch.no_grad()
def invert_batch(
    pipe,
    start_latents: torch.Tensor,
    prompts,
    start_index_for_sampling=10,
    guidance_scale=3.5,
    num_inference_steps=50,
    do_classifier_free_guidance=True,
    # negative_prompt="",
    device="cuda",
):
    text_embeddings = pipe._encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        # negative_prompt=negative_prompt,
    )

    latents = start_latents.clone()

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timesteps = list(reversed(timesteps))

    for i in range(num_inference_steps - 1 - start_index_for_sampling):
        t = timesteps[i]
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents, latents], dim=0)
        else:
            latent_model_input = latents

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # The typical forward step is:
        #   x_{t-1} = 1/sqrt(alpha_{t}) * ( x_{t} - (1-alpha_t)^{0.5} * e_theta(x_t, t) )
        #             +  (1 - alpha_{t-1})^{0.5} * e_theta(x_{t}, t)

        next_t = timesteps[i + 1] if (i + 1) < len(timesteps) else t
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        alpha_next_t = pipe.scheduler.alphas_cumprod[next_t]

        # Recompute latents by the "inverted" formula:
        latents = (
            (latents - (1 - alpha_t).sqrt() * noise_pred)
            * (alpha_next_t.sqrt() / alpha_t.sqrt())
        ) + (1 - alpha_next_t).sqrt() * noise_pred

    return latents


@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    # negative_prompt="",
    device="cuda",
):
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        # negative_prompt,
    )

    batch_size = text_embeddings.shape[0]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    if start_latents is None:
        start_latents = torch.randn(batch_size, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in range(start_step, num_inference_steps):
        t = pipe.scheduler.timesteps[i]

        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2, dim=0)
        else:
            latent_model_input = latents

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        prev_t = max(1, t.item() - (1000 // num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    latents = 1 / pipe.vae.config.scaling_factor * latents
    images = pipe.vae.decode(latents).sample  # tensor in [-1, 1]
    images = (images / 2 + 0.5).clamp(0, 1)  # now in [0,1]
    return images


def init_pipeline(model_id, dtype, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    scheduler_config = pipe.scheduler.config
    # scheduler_config["final_sigmas_type"] = "sigma_min"
    # scheduler_config["algorithm_type"] = "dpmsolver++"
    pipe.scheduler = DDIMScheduler.from_config(scheduler_config)

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    return pipe


def compute_clip_score_all(gen_images, prompts, device="cuda", batch_size=32):
    n = len(prompts)
    scores = []

    # If in [0,1], convert to [0,255]:
    if gen_images.max() <= 1.0:
        gen_images = (gen_images * 255.0).clamp(0, 255).to(torch.uint8)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = gen_images[start:end]  # (B, 3, H, W)
        batch_prompts = prompts[start:end]
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.inference_mode():
            batch_score = clip_score(
                batch_imgs,
                batch_prompts,
                model_name_or_path="openai/clip-vit-base-patch16",
            )
        scores.append(batch_score.mean().item())

    return float(sum(scores) / len(scores))


def compute_fid_torchmetrics(real_images, gen_images, device="cuda", batch_size=32):
    fid = FrechetInceptionDistance().to(device)
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
        fid.update(real_batch, real=True)
        fid.update(gen_batch, real=False)

    return fid.compute().item()


def preprocess_image(img: Image.Image, size=(512, 512), dtype=torch.float16):
    img = resize(img, size)
    img = to_tensor(img)
    if dtype == torch.float16:
        img = img.half()
    return img


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

    print("Loading MS-COCO test set (bitmind/MS-COCO)...")
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

    print("Initializing pipeline...")
    pipe = init_pipeline(args.model_id, dtype, device)

    rm_model = RM.load("ImageReward-v1.0")

    real_images_list = []
    rec_images_list = []
    prompts_list = []

    latent_mse_vals = []
    rec_mse_vals = []
    image_reward_vals = []

    dataloader_iter = range(0, len(dataset), args.batch_size)
    for start_i in tqdm(dataloader_iter, desc="Evaluation Loop"):
        end_i = min(start_i + args.batch_size, len(dataset))
        batch = dataset[start_i:end_i]

        pil_images = []
        prompts = []
        for i in range(end_i - start_i):
            pil_images.append(batch["image"][i].convert("RGB"))
            prompts.append(batch["sentences"][i]["raw"])

        real_imgs = []
        for img in pil_images:
            real_imgs.append(preprocess_image(img, size=(512, 512), dtype=dtype))
        real_imgs = torch.stack(real_imgs, dim=0).to(device)  # (B,3,512,512)

        with torch.no_grad():
            latents = (
                pipe.vae.encode((real_imgs * 2.0 - 1.0)).latent_dist.sample() * 0.18215
            )

        inverted_latents = invert_batch(
            pipe,
            start_latents=latents,
            start_index_for_sampling=25,
            prompts=prompts,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inversion_steps,
            device=device,
        )

        generated_images = sample(
            pipe=pipe,
            prompt=prompts,
            start_latents=inverted_latents,
            start_step=25,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inversion_steps,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            # negative_prompt="",
            device=device,
        )

        rec_images = pipe.numpy_to_pil(
            generated_images.cpu().permute(0, 2, 3, 1).float().numpy()
        )

        mse_lat = F.mse_loss(inverted_latents, latents).item()
        latent_mse_vals.append(mse_lat)

        mse_img = F.mse_loss(generated_images, real_imgs).item()
        rec_mse_vals.append(mse_img)

        batch_ir_scores = []
        for prompt_i, pil_img in zip(prompts, rec_images):
            score_i = rm_model.score(prompt_i, pil_img)
            batch_ir_scores.append(score_i)
        avg_batch_ir = float(np.mean(batch_ir_scores))
        image_reward_vals.append(avg_batch_ir)

        real_images_list.append(real_imgs.cpu())
        rec_images_list.append(generated_images)
        prompts_list.extend(prompts)

        del real_imgs, generated_images, latents, inverted_latents, rec_images
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
    print(f"Mean Latent MSE: {mean_latent_mse:.6f}")

    mean_rec_mse = float(np.mean(rec_mse_vals))
    print(f"Mean Reconstruction MSE (pixel space): {mean_rec_mse:.6f}")

    mean_ir_score = float(np.mean(image_reward_vals))
    print(f"Mean ImageReward Score: {mean_ir_score:.4f}")


if __name__ == "__main__":
    main()
