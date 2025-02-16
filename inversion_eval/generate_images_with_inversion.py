import os
import gc
import random
import argparse
import io
import torch

from PIL import Image
from tqdm import tqdm
import numpy as np

from datasets import load_dataset

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
from torchvision.transforms.functional import to_tensor, resize

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
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
        default=100,
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
    start_index_for_sampling=0,
    guidance_scale=3.5,
    num_inference_steps=50,
    do_classifier_free_guidance=True,
    device="cuda",
):
    text_embeddings = pipe._encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    latents = start_latents.clone()

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timesteps = list(reversed(timesteps))

    for i in tqdm(
        range(num_inference_steps - 1 - start_index_for_sampling), desc="Inverting"
    ):
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
    device="cuda",
):
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    )
    batch_size = text_embeddings.shape[0]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    if start_latents is None:
        start_latents = torch.randn(batch_size, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps), desc="Sampling"):
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

        # Manual DDIM update:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    latents = 1 / pipe.vae.config.scaling_factor * latents
    images = pipe.vae.decode(latents).sample  # returns tensor in [-1, 1]
    images = (images / 2 + 0.5).clamp(0, 1)  # now in [0, 1]
    return images  # tensor shape: [B, 3, H, W]


def init_pipeline(model_id, dtype, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    scheduler_config = pipe.scheduler.config
    # Replace with a DDIMScheduler (you can adjust parameters as needed)
    pipe.scheduler = DDIMScheduler.from_config(scheduler_config)

    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    return pipe


def preprocess_image(img: Image.Image, size=(512, 512), dtype=torch.float16):
    img = resize(img, size)
    img = to_tensor(img)
    if dtype == torch.float16:
        img = img.half()
    return img


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading MS-COCO validation set (bitmind/MS-COCO)...")
    data_files = {
        "test": "data/test-*-of-*.parquet",
    }
    dataset = load_dataset(
        "bitmind/MS-COCO",
        data_files=data_files,
        split="test",
        verification_mode="no_checks",
    )
    print("Total samples in val split:", len(dataset))

    sample_size = min(args.sample_size, len(dataset))
    subset_indices = random.sample(range(len(dataset)), sample_size)
    dataset = dataset.select(subset_indices)
    print("Initializing pipeline...")
    pipe = init_pipeline(args.model_id, dtype, device)
    # pipe.scheduler.num_inference_steps = args.num_inference_steps

    print(f"Generating images for {len(dataset)} prompts...")
    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size), desc="Generating images"
    ):
        end_i = min(start_idx + args.batch_size, len(dataset))
        batch = dataset[start_idx:end_i]

        pil_images = []
        prompts = []
        for i in range(args.batch_size):
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
            start_step=25,
            start_latents=inverted_latents,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inversion_steps,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            device=device,
        )

        gen_pil_images = pipe.numpy_to_pil(
            generated_images.cpu().permute(0, 2, 3, 1).float().numpy()
        )

        for idx, pil_img in enumerate(gen_pil_images):
            save_path = os.path.join(
                args.output_dir, f"generated_{start_idx + idx:06d}.png"
            )
            pil_img.save(save_path)
            pil_images[idx].save(
                os.path.join(args.output_dir, f"real_{start_idx + idx:06d}.png")
            )

        del (
            batch,
            generated_images,
            prompts,
            pil_images,
            gen_pil_images,
            inverted_latents,
            latents,
            real_imgs,
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Image generation complete. Saved images to: {args.output_dir}")


if __name__ == "__main__":
    main()
