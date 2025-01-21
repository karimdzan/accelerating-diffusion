import torch
from diffusers import (
    DDPMPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
)
from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from piq import FID
import matplotlib.pyplot as plt
import gc


class SingleImageDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return {"images": self.tensor[idx]}


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("pcuenq/lsun-bedrooms", split="test")
num_reference_images = len(dataset)

transform = Compose(
    [
        Resize((256, 256)),
        ToTensor(),
    ]
)

pipeline = DDPMPipeline.from_pretrained(
    "google/ddpm-bedroom-256", torch_dtype=torch.float16
).to(device)
pipeline.set_progress_bar_config(disable=True)

nfes = {
    "multi_step": [10, 12, 15, 20, 25],
    "single_step": [10, 12, 15, 20, 25],
    "ddim": [10, 12, 15, 20, 50, 100],
}

schedulers = {
    "multi_step": DPMSolverMultistepScheduler,
    "single_step": DPMSolverSinglestepScheduler,
    "ddim": DDIMScheduler,
}

results = {}

batch_size = 64
num_batches = num_reference_images // batch_size
remaining_images = num_reference_images % batch_size

fid = FID()

for scheduler_name, scheduler_cls in schedulers.items():
    print(f"Evaluating {scheduler_name} scheduler...")
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

    if scheduler_name in ["multi_step", "single_step"]:
        pipeline.scheduler.solver_order = 2

    scheduler_results = []

    for nfe in tqdm(nfes[scheduler_name], desc=f"{scheduler_name} evaluations"):
        pipeline.scheduler.num_inference_steps = nfe

        img_tensor = []
        gen_tensor = []

        for batch_idx in tqdm(range(num_batches)):
            batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size][
                "image"
            ]
            transformed_batch = torch.stack([transform(img) for img in batch])
            img_tensor.append(transformed_batch)

            with torch.no_grad():
                generated_imgs = pipeline(
                    batch_size=batch_size, num_inference_steps=nfe
                ).images

            transformed_gen_batch = torch.stack(
                [transform(img) for img in generated_imgs]
            )
            gen_tensor.append(transformed_gen_batch)

            torch.cuda.empty_cache()
            gc.collect()

        if remaining_images > 0:
            batch = dataset[-remaining_images:]["image"]
            transformed_batch = torch.stack([transform(img) for img in batch])
            img_tensor.append(transformed_batch)

            with torch.no_grad():
                generated_imgs = pipeline(
                    batch_size=remaining_images, num_inference_steps=nfe
                ).images
            transformed_gen_batch = torch.stack(
                [transform(img) for img in generated_imgs]
            )
            gen_tensor.append(transformed_gen_batch)

            torch.cuda.empty_cache()
            gc.collect()

        img_tensor = torch.cat(img_tensor, dim=0)
        gen_tensor = torch.cat(gen_tensor, dim=0)

        img_loader = DataLoader(SingleImageDataset(img_tensor), batch_size=batch_size)
        gen_loader = DataLoader(SingleImageDataset(gen_tensor), batch_size=batch_size)

        img_feats = fid.compute_feats(img_loader)
        gen_feats = fid.compute_feats(gen_loader)
        fid_score = fid(img_feats, gen_feats)

        print(f"NFE: {nfe}, FID: {fid_score:.2f}")
        scheduler_results.append((nfe, fid_score))

        del img_tensor, gen_tensor, img_feats, gen_feats
        torch.cuda.empty_cache()
        gc.collect()

    results[scheduler_name] = scheduler_results

data = {}
for scheduler_name, scheduler_results in results.items():
    print(f"Results for {scheduler_name} scheduler:")
    nfes = []
    fids = []
    for nfe, fid in scheduler_results:
        nfes.append(nfe)
        fids.append(fid)
        print(f"NFE: {nfe}, FID: {fid:.2f}")
    data[scheduler_name] = {"NFE": nfes, "FID": fids}

plt.figure(figsize=(8, 6))
for scheduler, results in data.items():
    plt.plot(results["NFE"], results["FID"], marker="o", label=scheduler)

plt.xlabel("NFE (Number of Function Evaluations)", fontsize=12)
plt.ylabel("FID (Frechet Inception Distance)", fontsize=12)
plt.title("FID vs. NFE for Different Schedulers", fontsize=14)
plt.legend(title="Schedulers", fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.show()
