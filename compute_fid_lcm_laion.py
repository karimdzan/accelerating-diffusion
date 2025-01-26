import gc
import io
import os
import pickle
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from piq import FID
from piq.feature_extractors import InceptionV3
from tqdm import tqdm
from PIL import Image
import argparse

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0

from torchmetrics.functional.multimodal import clip_score
from functools import partial

os.environ["PYTORCH_SDP_ENABLE_BACKWARD_OPTIMIZATION"] = "1"
torch.backends.cudnn.benchmark = True

clip_score_fn = partial(
    clip_score,
    model_name_or_path="openai/clip-vit-base-patch16",
)


def init_pipeline():
    """Initialize the SDXL + LCM pipeline with the existing model & settings."""
    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl",
        torch_dtype=DTYPE,
        variant="fp16",
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet,
        torch_dtype=DTYPE,
        variant="fp16",
    ).to(DEVICE)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    pipe.safety_checker = None

    pipe.image_processor.do_rescale = False

    pipe.set_progress_bar_config(disable=True)
    return pipe


def preprocess_single_image(image_bytes):
    """Convert raw image bytes -> Torch tensor in [0,1], shape (3, 512, 512)."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
        ]
    )
    return tform(pil_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID & CLIP for LCM")
    parser.add_argument(
        "-i", "--index", type=int, default=2, help="Task index (1-based)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Batch size for processing (default: 6)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for tensors (default: float16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../generation_checkpoint.pt",
        help="Path to save/load checkpoints (default: ../generation_checkpoint.pt)",
    )
    args = parser.parse_args()

    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = DTYPE_MAP[args.dtype]

    BATCH_SIZE = args.batch_size
    DTYPE = dtype
    DEVICE = args.device
    CHECKPOINT_PATH = args.checkpoint_path

    n = 2**args.index

    print("Loading LAION dataset (512x512 resized)...")
    dataset = load_from_disk("../resized_laion_512x512")
    data_len = len(dataset)

    print("Initializing pipeline...")
    pipe = init_pipeline()

    inception_model = InceptionV3().to(DEVICE).eval()

    real_features_list = []
    gen_features_list = []

    clip_scores_list = []

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint = pickle.load(f)
        start_idx = checkpoint["index"]
        real_features_list = checkpoint["real_feats_list"]
        gen_features_list = checkpoint["gen_feats_list"]
        clip_scores_list = checkpoint["clip_scores_list"]

        print(f"Resuming from checkpoint at index {start_idx}")
        del checkpoint
    else:
        start_idx = 0

    torch.cuda.empty_cache()
    gc.collect()

    print("Starting generation + evaluation loop...")
    try:
        with torch.inference_mode(), torch.no_grad():
            for i in tqdm(range(start_idx, data_len, BATCH_SIZE), desc="Generating"):
                subset = dataset.select(range(i, min(i + BATCH_SIZE, data_len)))
                real_image_tensors = []
                prompts = []

                for sample in subset:
                    real_image_tensors.append(
                        preprocess_single_image(sample["image"]["bytes"])
                    )
                    prompts.append(sample["text"] or "")

                # (B, 3, 512, 512)
                real_images = torch.stack(real_image_tensors, dim=0).to(DEVICE)
                B_current = real_images.shape[0]

                out_images = pipe(
                    prompt=prompts,
                    num_inference_steps=n,
                    guidance_scale=8.0,
                    num_images_per_prompt=1,
                    height=512,
                    width=512,
                ).images

                gen_tensors = []
                for img_pil in out_images:
                    gen_tensors.append(T.ToTensor()(img_pil))
                gen_images = torch.stack(gen_tensors, dim=0).to(DEVICE)

                clip_batch_score = clip_score_fn(gen_images, prompts)  # [B]
                clip_scores_list.append(clip_batch_score.mean().item())

                # (B, dims)
                real_feats = inception_model((real_images))[0]
                gen_feats = inception_model((gen_images))[0]

                real_features_list.append(real_feats.cpu())
                gen_features_list.append(gen_feats.cpu())

                del real_images, gen_images, subset, out_images
                torch.cuda.empty_cache()
                gc.collect()

                # 6) (Optional) checkpoint every so often
                #    e.g., every 200 steps or if near the end
                #    Below is a simple example (uncomment if you want frequent checkpointing).
                # if (i // BATCH_SIZE) % 200 == 0 or (i + BATCH_SIZE >= data_len):
                #     with open(CHECKPOINT_PATH, "wb") as f:
                #         pickle.dump(
                #             {
                #                 "index": i + B_current,  # next index
                #                 "real_feats_list": real_features_list,
                #                 "gen_feats_list": gen_features_list,
                #                 "clip_scores_list": clip_scores_list,
                #             },
                #             f,
                #             protocol=pickle.HIGHEST_PROTOCOL,
                #         )

    except Exception as e:
        print(f"Interrupted, saving checkpoint. Error: {str(e)}")
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(
                {
                    "index": i,
                    "real_feats_list": real_features_list,
                    "gen_feats_list": gen_features_list,
                    "clip_scores_list": clip_scores_list,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        exit()

    # with open(CHECKPOINT_PATH, "wb") as f:
    #     pickle.dump(
    #         {
    #             "index": data_len,
    #             "real_feats_list": real_features_list,
    #             "gen_feats_list": gen_features_list,
    #             "clip_scores_list": clip_scores_list,
    #         },
    #         f,
    #         protocol=pickle.HIGHEST_PROTOCOL,
    #     )

    print("All generations complete. Computing final metrics...")

    avg_clip_score = sum(clip_scores_list) / len(clip_scores_list)

    print(f"Final average CLIP Score: {avg_clip_score:.4f}")

    real_feats = torch.cat(real_features_list, dim=0).squeeze(-1).squeeze(-1)
    gen_feats = torch.cat(gen_features_list, dim=0).squeeze(-1).squeeze(-1)

    fid_metric = FID().to(DEVICE)

    real_feats = real_feats.to(DEVICE)
    gen_feats = gen_feats.to(DEVICE)
    fid_score = fid_metric(real_feats, gen_feats)

    print(f"Final FID Score: {fid_score:.2f}")
