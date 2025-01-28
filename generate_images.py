import os
import gc
import random
import argparse
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from datasets import load_from_disk

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    DiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images and save to disk")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../laion_improved_aesthetics_6.5plus_with_images",
        help="Path to the LAION aesthetics dataset with images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20000,
        help="Number of random samples to generate",
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
        help="Device to use for generation",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=2,
        help="Exponent for num_inference_steps = 2**index",
    )
    return parser.parse_args()


def init_pipeline(dtype, device):
    # unet = UNet2DConditionModel.from_pretrained(
    #     "latent-consistency/lcm-sdxl", torch_dtype=dtype, variant="fp16"
    # )
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     unet=unet,
    #     torch_dtype=dtype,
    #     variant="fp16",
    # ).to(device)
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", torch_dtype=dtype
    ).to(device)

    # pipe.unet = torch.compile(pipe.unet, mode="max-autotune")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    pipe.set_progress_bar_config(disable=True)

    return pipe


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
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("Loading original LAION dataset...")
    dataset = load_from_disk(args.dataset_path)["train"]
    dataset_len = len(dataset)
    print(f"Total dataset size: {dataset_len}")

    print(f"Selecting {sample_size} random samples...")
    indices = random.sample(range(dataset_len), sample_size)
    subset = dataset.select(indices)
    del dataset
    gc.collect()

    prompts_list = []
    print("Gathering prompts...")
    for sample in tqdm(subset, desc="Gathering prompts"):
        prompts_list.append(sample["text"] or "")

    del subset
    gc.collect()

    print("Loading pipeline...")
    pipe = init_pipeline(dtype, device)

    print(f"Generating images for {sample_size} prompts...")
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
                # num_images_per_prompt=1,
            ).images

        for idx, img in enumerate(out_pil_images):
            save_path = os.path.join(output_dir, f"generated_{start_idx + idx:06d}.png")
            img.save(save_path)

        del out_pil_images
        gc.collect()

    print(f"Image generation complete. Saved images to: {output_dir}")


if __name__ == "__main__":
    main()
