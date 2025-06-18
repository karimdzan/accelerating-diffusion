from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import ImageReward as RM 
from src.config import RunConfig  
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler  
from src.ddim_scheduler import MyDDIMScheduler 
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline  
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image 
from diffusers.utils.torch_utils import randn_tensor  


def parse_args():
    p = argparse.ArgumentParser("NR‑based evaluation on MS‑COCO", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sample_size", type=int, default=1000)
    p.add_argument("--scheduler", choices=["euler", "ddim"], default="euler")
    p.add_argument("--data_root", default="data/test-*-of-*.parquet", help="Globbing expression for parquet shards")
    p.add_argument("--log_dir", default="logs_nr", help="Where to dump images & txt logs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=19.0,
        help="If set to -1, we match forward cons latents in reverse.",
    )
    p.add_argument("--nfe", type=int, default=50)

    return p.parse_args()


def centre_crop_resize(img: Image.Image, size=(512, 512)) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    # img = img.crop((left, top, left + s, top + s)).resize(size, Image.Resampling.LANCZOS)
    img = img.resize(size, Image.Resampling.LANCZOS)
    return img.convert("RGB")


def prepare_noise_list(device: torch.device, n_steps: int) -> List[torch.Tensor]:
    g = torch.Generator(device).manual_seed(7865)
    lat_shape = (1, 4, 64, 64)  
    return [randn_tensor(lat_shape, dtype=torch.float16, device=device, generator=g) for _ in range(n_steps)]


def invert_batch(pipe_inv: SDXLDDIMPipeline, images: List[Image.Image], prompts: List[str], cfg: RunConfig) -> Tuple[np.ndarray, torch.Tensor, List[torch.Tensor]]:
    out = pipe_inv(
        pipe_inv.device,
        prompt=prompts,
        image=images,
        num_inversion_steps=cfg.num_inversion_steps,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=0.0,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        inv_hp=[2, 0.1 * 4 * 64 * 64, 0.2, pipe_inv.cfg.scheduler_type],
        callback_on_step_end=lambda *a, **k: k,  
    )
    recon_imgs = out[0][0]
    return recon_imgs


def generate_batch(pipe_gen, start_latent: torch.Tensor, prompts: List[str], cfg: RunConfig):
    gen = pipe_gen(
        prompt=prompts,
        image=start_latent.to(pipe_gen.device),
        num_inference_steps=cfg.num_inference_steps,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        guidance_scale=cfg.guidance_scale,
        return_type="numpy",
    )
    imgs_np = np.stack([img for img in gen.images])  
    imgs_np = imgs_np.astype("float32") / 255.0  
    return imgs_np  


def main():
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print("Loading SDXL‑Turbo pipelines …")
    sched_cls = MyEulerAncestralDiscreteScheduler if args.scheduler == "euler" else MyDDIMScheduler

    pipe_inv = SDXLDDIMPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True, safety_checker=None
    ).to(device, dtype)
    pipe_gen = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True, safety_checker=None
    ).to(device, dtype)

    pipe_inv.scheduler = sched_cls.from_config(pipe_inv.scheduler.config)
    pipe_gen.scheduler = sched_cls.from_config(pipe_gen.scheduler.config)
    pipe_inv.scheduler.use_karras_sigmas = True
    pipe_gen.scheduler.use_karras_sigmas = True
    pipe_gen.set_progress_bar_config(disable=True)
    pipe_inv.set_progress_bar_config(disable=True)

    cfg = RunConfig(num_inference_steps=50, num_inversion_steps=args.nfe, guidance_scale=args.guidance_scale, inversion_max_step=0.6)
    pipe_inv.cfg = cfg
    pipe_inv.cfg.scheduler_type = args.scheduler

    noise_list = prepare_noise_list(device, cfg.num_inversion_steps)
    if hasattr(pipe_inv.scheduler, "set_noise_list"):
        pipe_inv.scheduler.set_noise_list(noise_list)
        pipe_gen.scheduler.set_noise_list(noise_list)

    print("Loading metrics …")
    fid = FrechetInceptionDistance().to(device)
    clip_scores, ir_scores, pix_mses = [], [], []
    rm_model = RM.load("ImageReward-v1.0")
    
    print("Loading dataset MS-COCO …")
    ds = load_dataset("bitmind/MS-COCO", data_files={"test": args.data_root}, split="test", verification_mode="no_checks")
    ds = ds.select(random.sample(range(len(ds)), min(args.sample_size, len(ds))))

    print("Starting evaluation process …")
    step = 0
    for idx in tqdm(range(0, len(ds), args.batch_size), desc="Batches"):
        batch = ds[idx : idx + args.batch_size]
        pil_imgs = [centre_crop_resize(im) for im in batch["image"]]
        prompts = [s["raw"] for s in batch["sentences"]]

        lat_final = invert_batch(pipe_inv, pil_imgs, prompts, cfg)
        gen_np = generate_batch(pipe_gen, lat_final, prompts, cfg)

        to_tensor = T.ToTensor()
        real_imgs_tensor = torch.stack([to_tensor(im) for im in pil_imgs])  # shape: [B, C, H, W]
        real_imgs_uint8 = (real_imgs_tensor * 255).clip(0, 255).to(torch.uint8).to(device)
        fake_imgs_tensor = torch.from_numpy(gen_np * 255).clip(0, 255).permute(0, 3, 1, 2).to(torch.uint8)

        fid.update(real_imgs_uint8, real=True)
        fid.update(fake_imgs_tensor.to(device), real=False)

        clip_scores.append(clip_score(fake_imgs_tensor, prompts).item())
        ir_scores.extend([rm_model.score(p, Image.fromarray((g*255).astype(np.uint8))) for p, g in zip(prompts, gen_np)])

        pixel_mse = F.mse_loss(fake_imgs_tensor, real_imgs_tensor).item()
        # print(f"[step {step}] pixel‑MSE = {pixel_mse:.4e}  CLIP = {clip_scores[-1]:.4f}  IR = {ir_scores[-1]:.4f}")
        pix_mses.append(pixel_mse)
        step += 1
        gc.collect(); torch.cuda.empty_cache()
        if step % args.log_every == 0:
            Path(args.log_dir).mkdir(exist_ok=True)
            Image.fromarray((gen_np[0]*255).astype(np.uint8)).save(Path(args.log_dir)/f"gen_{step}.jpg")
            batch['image'][0].save(Path(args.log_dir)/f"orig_{step}.jpg")
            print(f"  ↳ dumped sample images at step {step}")

    print("\n======= FINAL =======")
    print(f"FID         : {fid.compute().item():.4f}")
    print(f"mean CLIP   : {float(np.mean(clip_scores)):.4f}")
    print(f"mean IR     : {float(np.mean(ir_scores)):.4f}")
    print(f"mean Pixel MSE     : {float(np.mean(pix_mses)):.4f}")


if __name__ == "__main__":
    main()
