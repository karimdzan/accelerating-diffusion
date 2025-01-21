import gc
import io
import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from piq import FID
from tqdm import tqdm
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0


os.environ["PYTORCH_SDP_ENABLE_BACKWARD_OPTIMIZATION"] = "1"

torch.backends.cudnn.benchmark = True

BATCH_SIZE = 4
DTYPE = torch.float16
DEVICE = "cuda"
CHECKPOINT_PATH = "../generation_checkpoint.pt"

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

pipe.set_progress_bar_config(disable=True)

to_tensor = T.ToTensor()


def preprocess_dataset(dataset):
    tform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
        ]
    )
    all_tensors = []
    for img_dict in tqdm(dataset["image"], desc="Preprocessing Dataset"):
        with io.BytesIO(img_dict["bytes"]) as buffer:
            pil_img = Image.open(buffer).convert("RGB")
            all_tensors.append(tform(pil_img))

    return torch.stack(all_tensors, dim=0)  # (N, 3, 512, 512)


class TensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return {"images": self.tensor[idx]}


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_from_disk("../resized_laion_512x512")

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        gen_tensor_list = checkpoint["tensor"]
        start_idx = checkpoint["index"]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        gen_tensor_list = []
        start_idx = 0

    # print("Generating images...")
    # try:
    #     with torch.inference_mode(), torch.no_grad():
    #         for i in tqdm(
    #             range(start_idx, len(dataset), BATCH_SIZE), desc="Generating"
    #         ):
    #             out_images = pipe(
    #                 prompt="",
    #                 num_inference_steps=1,
    #                 guidance_scale=1.0,
    #                 num_images_per_prompt=BATCH_SIZE,
    #                 height=512,
    #                 width=512,
    #             ).images

    #             gen_batch = torch.stack([to_tensor(img) for img in out_images]).to(
    #                 DEVICE, DTYPE
    #             )
    #             gen_tensor_list.append(gen_batch)

    #             if (i // BATCH_SIZE) % 1250 == 0 or (i + BATCH_SIZE) >= len(dataset):
    #                 torch.save(
    #                     {"tensor": gen_tensor_list, "index": i + BATCH_SIZE},
    #                     CHECKPOINT_PATH,
    #                 )

    #             del out_images, gen_batch

    #             if (i // BATCH_SIZE) % 50 == 0:
    #                 torch.cuda.empty_cache()
    #             gc.collect()

    # except Exception as e:
    #     print(f"Interrupted, saving checkpoint. Error: {str(e)}")
    #     torch.save({"tensor": gen_tensor_list, "index": i}, CHECKPOINT_PATH)
    #     exit()

    # print("Concatenating generated images...")
    # gen_tensor = torch.cat(gen_tensor_list, dim=0)

    print("Preprocessing real images...")
    img_tensor = preprocess_dataset(dataset)
    # del dataset, gen_tensor_list
    # gc.collect()
    print(img_tensor.shape)
    real_loader = DataLoader(
        TensorDataset(img_tensor),
        batch_size=BATCH_SIZE * 2,
        num_workers=0,
    )
    gen_loader = DataLoader(
        TensorDataset(gen_tensor_list),
        batch_size=BATCH_SIZE * 2,
        num_workers=0,
    )

    print("Computing FID...")
    fid = FID().to(DEVICE)
    with torch.cuda.amp.autocast():
        real_feats = fid.compute_feats(real_loader)
        gen_feats = fid.compute_feats(gen_loader)
        fid_score = fid(real_feats, gen_feats)

    print(f"FID Score (1 step): {fid_score:.2f}")
