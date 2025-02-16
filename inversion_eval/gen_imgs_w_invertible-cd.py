import os
import gc
import random
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from diffusers import LCMScheduler, DDPMScheduler, StableDiffusionPipeline
import ImageReward as RM
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor, resize
from diffusers.models.attention_processor import AttnProcessor2_0

from utils.loading import load_models
from utils import p2p, generation, inversion

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

NUM_REVERSE_CONS_STEPS = 4
REVERSE_TIMESTEPS = [259, 519, 779, 999]
NUM_FORWARD_CONS_STEPS = 4
FORWARD_TIMESTEPS = [19, 259, 519, 779]
NUM_DDIM_STEPS = 50
START_TIMESTEP = 19


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iCD Model on MS-COCO")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=25010)
    parser.add_argument("--guidance_scale", type=float, default=19.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--forward_checkpoint", type=str, required=True)
    parser.add_argument("--reverse_checkpoint", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    return parser.parse_args()


def initialize_models(args):
    ldm_stable, reverse_cons_model, forward_cons_model = load_models(
        model_id="sd-legacy/stable-diffusion-v1-5",
        device=args.device,
        forward_checkpoint=args.forward_checkpoint,
        reverse_checkpoint=args.reverse_checkpoint,
        r=64,
        w_embed_dim=512,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    ldm_stable.unet.set_attn_processor(AttnProcessor2_0())
    reverse_cons_model.unet.set_attn_processor(AttnProcessor2_0())
    forward_cons_model.unet.set_attn_processor(AttnProcessor2_0())

    ldm_stable.set_progress_bar_config(disable=True)
    reverse_cons_model.set_progress_bar_config(disable=True)
    forward_cons_model.set_progress_bar_config(disable=True)

    ldm_stable.safety_checker = None
    reverse_cons_model.safety_checker = None
    forward_cons_model.safety_checker = None

    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )

    solver = generation.Generator(
        model=ldm_stable,
        noise_scheduler=noise_scheduler,
        n_steps=NUM_DDIM_STEPS,
        forward_cons_model=forward_cons_model,
        forward_timesteps=FORWARD_TIMESTEPS,
        reverse_cons_model=reverse_cons_model,
        reverse_timesteps=REVERSE_TIMESTEPS,
        num_endpoints=NUM_REVERSE_CONS_STEPS,
        num_forward_endpoints=NUM_FORWARD_CONS_STEPS,
        max_forward_timestep_index=49,
        start_timestep=START_TIMESTEP,
    )

    # Configure P2P components
    p2p.NUM_DDIM_STEPS = NUM_DDIM_STEPS
    p2p.tokenizer = ldm_stable.tokenizer
    p2p.device = args.device

    return ldm_stable, solver, reverse_cons_model


def generate_images_batch(solver, reverse_cons_model, prompts, latents, args):
    images = []
    generator = torch.Generator(device=args.device).manual_seed(42)
    controller = p2p.AttentionStore()
    images, gen_latents = generation.runner(
        guidance_scale=args.guidance_scale,
        tau1=args.tau,
        tau2=args.tau,
        is_cons_forward=True,
        model=reverse_cons_model,
        dynamic_guidance=True,
        w_embed_dim=512,
        start_time=50,
        solver=solver,
        prompt=prompts,
        controller=controller,
        generator=generator,
        latent=latents,
        return_type="image",
        # num_inference_steps=50,
    )
    return images, gen_latents


def invert_images_batch(solver, prompts, images):
    (image_gt, image_rec), latent, uncond_embeddings, latent_orig = inversion.invert(
        # Playing params
        is_cons_inversion=True,
        # do_npi=False,
        # do_nti=False,
        w_embed_dim=512,
        stop_step=50,  # from [0, NUM_DDIM_STEPS]
        inv_guidance_scale=0.0,
        dynamic_guidance=False,
        tau1=0.0,
        tau2=0.0,
        # Fixed params
        solver=solver,
        images=images,
        prompt=prompts,
        # num_inner_steps=10,
        # early_stop_epsilon=1e-5,
        seed=42,
    )

    return image_gt, image_rec, latent, latent_orig


def main():
    args = parse_args()
    torch.manual_seed(42)

    ldm_stable, solver, reverse_cons_model = initialize_models(args)
    rm_model = RM.load("ImageReward-v1.0")

    # dataset = load_dataset(
    #     "bitmind/MS-COCO", split="test", verification_mode="no_checks"
    # )
    data_files = {
        "test": "data/test-*-of-*.parquet",
    }
    dataset = load_dataset(
        "bitmind/MS-COCO",
        data_files=data_files,
        split="test",
        verification_mode="no_checks",
    )
    dataset = dataset.select(
        random.sample(range(len(dataset)), min(args.sample_size, len(dataset)))
    )

    mse_latent, mse_real = [], []
    fid = FrechetInceptionDistance().to(args.device)
    clip_scores, ir_scores = [], []

    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size), desc="Processing batches"
    ):
        batch = dataset[start_idx : start_idx + args.batch_size]
        batch_images = [
            img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            for img in batch["image"]
        ]
        batch_prompts = [s["raw"] for s in batch["sentences"]]

        images_gt, images_rec, latents, latents_orig = invert_images_batch(
            solver, batch_prompts, batch_images
        )

        gen_images_batch, gen_latents_batch = generate_images_batch(
            solver, reverse_cons_model, batch_prompts, latents, args
        )

        with torch.no_grad():
            fid.update(torch.from_numpy(images_rec).to(args.device), real=True)
            fid.update(torch.from_numpy(gen_images_batch).to(args.device), real=False)

            clip_scores.append(
                clip_score(torch.from_numpy(gen_images_batch), batch_prompts).item()
            )

            pil_images = [
                T.ToPILImage()(img.transpose(1, 2, 0)) for img in gen_images_batch
            ]
            ir_scores.extend(
                [rm_model.score(p, img) for p, img in zip(batch_prompts, pil_images)]
            )

            for i in range(len(gen_images_batch)):
                generation.to_pil_images(gen_images_batch[i].transpose(1, 2, 0)).save(
                    f"test_inv+gen_{start_idx + i}_iCD-SD1.5.jpg"
                )

        mse_latent.append(
            F.mse_loss(
                torch.from_numpy(gen_images_batch).float(),
                torch.from_numpy(images_rec).float(),
            ).item()
        )
        mse_real.append(F.mse_loss(gen_latents_batch, latents_orig).item())

        del latents, gen_latents_batch, gen_images_batch, images_rec, images_gt
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFID: {fid.compute().item():.4f}")
    print(f"CLIP Score: {np.mean(clip_scores):.4f}")
    print(f"ImageReward: {np.mean(ir_scores):.4f}")

    print(f"Pixel MSE: {np.mean(mse_real):.4f}")
    print(f"Latent MSE: {np.mean(mse_latent):.4f}")


if __name__ == "__main__":
    main()
