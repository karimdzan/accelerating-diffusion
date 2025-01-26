import os
import gc
import random
import argparse
import io
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from datasets import load_from_disk
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0

os.environ["PYTORCH_SDP_ENABLE_BACKWARD_OPTIMIZATION"] = "1"
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID & CLIP for LCM (end-of-run computation)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../laion_improved_aesthetics_6.5plus_with_images",
        help="Path to the LAION aesthetics dataset with images",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20000,
        help="Number of random samples to evaluate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for *generation*"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for the pipeline and stored tensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation and metrics",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=2,
        help="Exponent for num_inference_steps = 2**index",
    )
    return parser.parse_args()


def init_pipeline(dtype, device):
    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl", torch_dtype=dtype, variant="fp16"
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet,
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.safety_checker = None
    pipe.image_processor.do_rescale = False
    pipe.set_progress_bar_config(disable=True)
    return pipe


def preprocess_image_bytes(image_bytes, dtype=torch.float16):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    out = tform(pil_img)
    if dtype == torch.float16:
        out = out.half()
    return out


def compute_clip_score_all(
    gen_images: torch.Tensor,
    prompts: list,
    device="cuda",
    batch_size=32,
) -> float:
    n = len(prompts)
    scores = []

    if gen_images.max() <= 1.0:
        gen_images = (gen_images * 255.0).clamp(0, 255).to(torch.uint8)

    for start in tqdm(range(0, n, batch_size), desc="Computing CLIP score"):
        end = min(start + batch_size, n)
        batch_imgs = gen_images[start:end]  # (B, 3, H, W)
        batch_prompts = prompts[start:end]
        batch_imgs = batch_imgs.clamp(0, 255).to(torch.uint8)

        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.inference_mode():
            batch_score = clip_score(
                batch_imgs,
                batch_prompts,
                model_name_or_path="openai/clip-vit-base-patch16",
            )
        scores.append(batch_score.mean().item())

        del batch_imgs, batch_prompts, batch_score
        torch.cuda.empty_cache()
        gc.collect()

    return float(sum(scores) / len(scores))


def compute_fid_torchmetrics(
    real_images: torch.Tensor, gen_images: torch.Tensor, device="cuda", batch_size=32
) -> float:
    fid = FrechetInceptionDistance().to(device)
    n = real_images.shape[0]
    for idx in tqdm(range(0, n, batch_size), desc="Computing FID"):
        end = min(idx + batch_size, n)
        real_batch = real_images[idx:end]
        gen_batch = gen_images[idx:end]
        if real_batch.max() <= 1.0:
            real_batch = (real_batch * 255.0).clamp(0, 255)
        if gen_batch.max() <= 1.0:
            gen_batch = (gen_batch * 255.0).clamp(0, 255)
        real_batch = real_batch.to(torch.uint8)
        gen_batch = gen_batch.to(torch.uint8)
        real_batch = real_batch.to(device, non_blocking=True)
        gen_batch = gen_batch.to(device, non_blocking=True)
        fid.update(real_batch, real=True)
        fid.update(gen_batch, real=False)

        del real_batch, gen_batch
        torch.cuda.empty_cache()
        gc.collect()

    return fid.compute().item()


def main():
    args = parse_args()
    if args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    device = args.device
    sample_size = args.sample_size
    batch_size = args.batch_size
    num_inference_steps = 2**args.index

    print("Loading original LAION dataset...")
    dataset = load_from_disk(args.dataset_path)["train"]
    dataset_len = len(dataset)
    print(f"Total dataset size: {dataset_len}")

    print(f"Selecting {sample_size} random samples...")
    indices = random.sample(range(dataset_len), sample_size)
    subset = dataset.select(indices)
    del dataset
    gc.collect()

    real_images_list = []
    prompts_list = []

    print("Preprocessing real images + gathering prompts in memory...")
    for sample in tqdm(subset, desc="Preprocessing"):
        img_tensor = preprocess_image_bytes(sample["image"]["bytes"], dtype=dtype)
        real_images_list.append(img_tensor)
        prompts_list.append(sample["text"] or "")

    real_images = torch.stack(real_images_list, dim=0)
    del real_images_list
    gc.collect()

    print("Loading pipeline...")
    pipe = init_pipeline(dtype, device)

    gen_images_list = []
    print("Generating images from text prompts...")
    N = len(prompts_list)
    for start_idx in tqdm(range(0, N, batch_size), desc="Generating images"):
        end_idx = min(start_idx + batch_size, N)
        batch_prompts = prompts_list[start_idx:end_idx]

        with torch.inference_mode(), torch.no_grad():
            out_pil_images = pipe(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
                height=512,
                width=512,
                num_images_per_prompt=1,
            ).images

        batch_gen_tensors = []
        for im in out_pil_images:
            t = T.ToTensor()(im)
            if dtype == torch.float16:
                t = t.half()
            batch_gen_tensors.append(t)

        gen_images_list.extend(batch_gen_tensors)
        del batch_gen_tensors, out_pil_images
        gc.collect()

    gen_images = torch.stack(gen_images_list, dim=0)
    del gen_images_list
    gc.collect()

    print(
        f"All images generated. Real shape: {real_images.shape}, Gen shape: {gen_images.shape}"
    )

    print("Computing final CLIP score (average over all samples)...")
    clip_score_avg = compute_clip_score_all(gen_images, prompts_list, batch_size=32)
    print(f"CLIP Score: {clip_score_avg:.4f}")

    print("Computing FID with torchmetrics...")
    fid_value = compute_fid_torchmetrics(real_images, gen_images, batch_size=32)
    print(f"FID Score: {fid_value:.2f}")


if __name__ == "__main__":
    main()
