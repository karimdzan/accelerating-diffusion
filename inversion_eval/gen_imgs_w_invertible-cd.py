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
from diffusers import DDPMScheduler
import ImageReward as RM
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor
from diffusers.models.attention_processor import AttnProcessor2_0

# NEW: import wandb for logging
import wandb

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
WANDB_TOKEN = os.environ["WANDB_TOKEN"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iCD Model on MS-COCO")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=25010)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=19.0,
        help="If set to -1, we match forward cons latents in reverse.",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--forward_checkpoint", type=str, required=True)
    parser.add_argument("--reverse_checkpoint", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs_training",
        help="Where to save example images & logs.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log every X iterations (e.g. images, partial metrics).",
    )
    return parser.parse_args()


def initialize_models(args):
    ldm_stable, reverse_cons_model, forward_cons_model = load_models(
        model_id="sd-legacy/stable-diffusion-v1-5",
        device=args.device,
        forward_checkpoint=args.forward_checkpoint,
        reverse_checkpoint=args.reverse_checkpoint,
        r=64,
        w_embed_dim=512,  # embedding dimension
        teacher_checkpoint=args.teacher_checkpoint,
    )
    # Use the newer attention processor to reduce VRAM usage
    ldm_stable.unet.set_attn_processor(AttnProcessor2_0())
    reverse_cons_model.unet.set_attn_processor(AttnProcessor2_0())
    forward_cons_model.unet.set_attn_processor(AttnProcessor2_0())

    ldm_stable.set_progress_bar_config(disable=True)
    reverse_cons_model.set_progress_bar_config(disable=True)
    forward_cons_model.set_progress_bar_config(disable=True)

    # Disable safety checker for now
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

    return ldm_stable, solver, reverse_cons_model, forward_cons_model


def generate_images_batch(solver, reverse_cons_model, prompts, latent, args):
    # Run the reversed or forward generation
    images = []
    generator = torch.Generator(device=args.device).manual_seed(42)
    controller = p2p.AttentionStore()

    images, gen_latent, latents = generation.runner(
        guidance_scale=args.guidance_scale,
        tau1=args.tau,
        tau2=args.tau,
        is_cons_forward=True,
        model=reverse_cons_model,
        dynamic_guidance=False,
        w_embed_dim=512,
        start_time=50,
        solver=solver,
        prompt=prompts,
        controller=controller,
        generator=generator,
        latent=latent,
        return_type="image",
    )

    return images, gen_latent, latents


def invert_images_batch(
    solver, forward_cons_model, prompts, images, use_reverse_model=False
):
    # Inversion using forward model or solver's approach
    (image_gt, image_rec), latents, uncond_embeddings, latent_orig = inversion.invert(
        is_cons_inversion=True,
        w_embed_dim=512,
        stop_step=50,
        inv_guidance_scale=0.0,  # purely unconditional forward pass
        dynamic_guidance=False,
        tau1=0.0,
        tau2=0.0,
        solver=solver,
        images=images,
        prompt=prompts,
        seed=42,
        use_reverse_model=use_reverse_model,
    )

    return image_gt, image_rec, latents[-1], latents, latent_orig


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    torch.manual_seed(42)

    # Initialize wandb and log the configuration
    wandb.login(key=WANDB_TOKEN)
    wandb.init(project="icd_evaluation", config=vars(args))
    ldm_stable, solver, reverse_cons_model, forward_cons_model = initialize_models(args)
    rm_model = RM.load("ImageReward-v1.0")

    # Initialize an optimizer specifically for solver.w_embedding
    optimizer = torch.optim.Adam([solver.w_embedding], lr=1e-4)

    # Prepare the dataset
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

    fid = FrechetInceptionDistance().to(args.device)
    clip_scores, ir_scores = [], []
    diff_latents = []
    mse_latent_log = []
    mse_real_log = []

    # We'll keep track of steps for logging
    step_counter = 0

    # Main training/inference loop
    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size), desc="Processing batches"
    ):
        batch = dataset[start_idx : start_idx + args.batch_size]
        batch_images = [
            img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            for img in batch["image"]
        ]
        batch_prompts = [s["raw"] for s in batch["sentences"]]

        # 1) Inversion to get latents from the forward (teacher) direction
        images_gt, images_rec, latent, latents1, latents_orig = invert_images_batch(
            solver,
            forward_cons_model,
            batch_prompts,
            batch_images,
            use_reverse_model=True,
        )

        # 2) "Reverse" generation (with or without guidance scale)
        gen_images_batch, gen_latents_batch, latents2 = generate_images_batch(
            solver,
            reverse_cons_model,
            batch_prompts,
            latent,
            args,
            step_idx=step_counter,
        )

        # Evaluate FID/CLIP/ImageReward on CPU-friendly copies
        with torch.no_grad():
            # Update FID stats
            fid.update(torch.from_numpy(images_rec).to(args.device), real=True)
            fid.update(torch.from_numpy(gen_images_batch).to(args.device), real=False)

            # CLIP Score
            clip_scores.append(
                clip_score(torch.from_numpy(gen_images_batch), batch_prompts).item()
            )

            # ImageReward
            pil_images = [
                T.ToPILImage()(img.transpose(1, 2, 0)) for img in gen_images_batch
            ]
            ir_scores.extend(
                [
                    rm_model.score(prompt, img)
                    for prompt, img in zip(batch_prompts, pil_images)
                ]
            )

        # 3) Basic MSE logs: pixel-level & latent-level
        pixel_mse = F.mse_loss(
            torch.from_numpy(gen_images_batch).float(),
            torch.from_numpy(images_rec).float(),
        ).item()
        latent_mse = F.mse_loss(gen_latents_batch, latents_orig).item()
        mse_real_log.append(pixel_mse)
        mse_latent_log.append(latent_mse)

        # 4) Prepare latents for training step
        a1 = latents1[1] - latents2[3]
        a2 = latents1[2] - latents2[2]
        a3 = latents1[3] - latents2[1]

        diff_latents.append((a1, a2, a3))

        # Use latents from forward pass vs. reverse pass (as an example)
        latent_forward1 = latents1[1].detach()  # no grad
        latent_reverse1 = latents2[3]  # reverse latent
        latent_forward2 = latents1[2].detach()  # no grad
        latent_reverse2 = latents2[2]  # reverse latent
        latent_forward3 = latents1[3].detach()  # no grad
        latent_reverse3 = latents2[1]  # reverse latent

        # 5) Compute MSE loss for training the embedding
        loss1 = F.mse_loss(latent_forward1, latent_reverse1)
        loss2 = F.mse_loss(latent_forward2, latent_reverse2)
        loss3 = F.mse_loss(latent_forward3, latent_reverse3)

        loss = loss1 + loss2 + loss3
        # 6) Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) Update step counter and log metrics to wandb
        step_counter += 1
        batch_metrics = {
            "step": step_counter,
            "batch_start": start_idx,
            "batch_end": start_idx + args.batch_size,
            "guidance_scale": args.guidance_scale,
            "loss": loss.item(),
            "loss_latent_r_1": loss1.item(),
            "loss_latent_r_2": loss2.item(),
            "loss_latent_r_3": loss3.item(),
            "pixel_mse": pixel_mse,
            "latent_mse": latent_mse,
            "diff_latents_a1": a1.mean().item(),
            "diff_latents_a2": a2.mean().item(),
            "diff_latents_a3": a3.mean().item(),
        }
        wandb.log(batch_metrics, step=step_counter)

        # 8) Periodically log sample images to wandb
        if step_counter % args.log_every == 0:
            try:
                # Convert images to PIL for logging
                rec_pil = T.ToPILImage()(images_rec[0].transpose(1, 2, 0))
                gen_pil = T.ToPILImage()(gen_images_batch[0].transpose(1, 2, 0))

                # Optionally, still save to disk if desired
                rec_pil.save(
                    os.path.join(args.log_dir, f"step_{step_counter}_sample_rec.jpg")
                )
                gen_pil.save(
                    os.path.join(args.log_dir, f"step_{step_counter}_sample_gen.jpg")
                )

                # Log images to wandb
                wandb.log(
                    {
                        "sample_rec": wandb.Image(
                            rec_pil,
                            caption=f"Reconstructed image at step {step_counter}",
                        ),
                        "sample_gen": wandb.Image(
                            gen_pil, caption=f"Generated image at step {step_counter}"
                        ),
                    },
                    step=step_counter,
                )
            except Exception as e:
                wandb.log(
                    {"image_logging_warning": f"Could not log sample images: {e}"},
                    step=step_counter,
                )

        # Cleanup
        del (
            latent,
            gen_latents_batch,
            gen_images_batch,
            images_rec,
            images_gt,
            latents1,
            latents2,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summaries after epoch ends ---
    d1, d2, d3 = 0, 0, 0
    for da1, da2, da3 in diff_latents:
        d1 += da1
        d2 += da2
        d3 += da3
    n = len(diff_latents) if len(diff_latents) > 0 else 1  # prevent division by zero
    avg_diff = {
        "avg_diff_latents_a1": d1 / n,
        "avg_diff_latents_a2": d2 / n,
        "avg_diff_latents_a3": d3 / n,
    }
    wandb.log(avg_diff, step=step_counter)
    print("\nAverage differences between the latents across entire dataset:")
    print(f"  - a1: {avg_diff['avg_diff_latents_a1']:.4f}")
    print(f"  - a2: {avg_diff['avg_diff_latents_a2']:.4f}")
    print(f"  - a3: {avg_diff['avg_diff_latents_a3']:.4f}")

    # Final metrics
    final_fid = fid.compute().item()
    mean_clip = np.mean(clip_scores)
    mean_ir = np.mean(ir_scores)
    final_metrics = {
        "final_fid": final_fid,
        "mean_clip_score": mean_clip,
        "mean_image_reward": mean_ir,
        "avg_latent_mse": np.mean(mse_latent_log),
        "avg_pixel_mse": np.mean(mse_real_log),
    }
    wandb.log(final_metrics, step=step_counter)
    print(f"\nFID: {final_fid:.4f}")
    print(f"CLIP Score: {mean_clip:.4f}")
    print(f"ImageReward: {mean_ir:.4f}")
    print(f"Latent MSE (average): {final_metrics['avg_latent_mse']:.4f}")
    print(f"Pixel MSE (average): {final_metrics['avg_pixel_mse']:.4f}")

    # Mark the run as finished in wandb
    wandb.finish()


if __name__ == "__main__":
    main()
