import torch
from diffusers import (
    DDPMPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
)
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from pytorch_fid import fid_score
import matplotlib.pyplot as plt


def compute_fid(generated_dir, reference_dir):
    return fid_score.calculate_fid_given_paths(
        [generated_dir, reference_dir], batch_size=64, device=device, dims=2048
    )


device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
pipeline.set_progress_bar_config(disable=True)

reference_dir = "cifar10_images"

output_dir = "dpm_solver_results"
os.makedirs(output_dir, exist_ok=True)

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

total_images = 10000
batch_size = 64
num_batches = total_images // batch_size

for scheduler_name, scheduler_cls in schedulers.items():
    print(f"Evaluating {scheduler_name} scheduler...")
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
    if scheduler_name in ["multi_step", "single_step"]:
        pipeline.scheduler.solver_order = 2

    scheduler_results = []

    for nfe in tqdm(nfes[scheduler_name], desc=f"{scheduler_name} evaluations"):
        pipeline.scheduler.num_inference_steps = nfe
        nfe_dir = os.path.join(output_dir, f"{scheduler_name}_nfe_{nfe}")
        os.makedirs(nfe_dir, exist_ok=True)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        image_counter = 0
        for batch_idx in range(num_batches):
            with torch.no_grad():
                images = pipeline(batch_size=batch_size, num_inference_steps=nfe).images

            for idx, img in enumerate(images):
                save_image(
                    ToTensor()(img),
                    os.path.join(nfe_dir, f"img_{image_counter}.png"),
                )
                image_counter += 1

        remaining_images = total_images % batch_size
        if remaining_images > 0:
            with torch.no_grad():
                images = pipeline(
                    batch_size=remaining_images, num_inference_steps=nfe
                ).images

            for idx, img in enumerate(images):
                save_image(
                    ToTensor()(img),
                    os.path.join(nfe_dir, f"img_{image_counter}.png"),
                )
                image_counter += 1

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        fid = compute_fid(nfe_dir, reference_dir)
        print(f"NFE: {nfe}, Time: {elapsed_time:.2f}s, FID: {fid:.2f}")

        scheduler_results.append((nfe, elapsed_time, fid))

    results[scheduler_name] = scheduler_results

data = {}
for scheduler_name, scheduler_results in results.items():
    print(f"Results for {scheduler_name} scheduler:")
    nfes = []
    fids = []
    for nfe, time, fid in scheduler_results:
        nfes.append(nfe)
        fids.append(fid)
        print(f"NFE: {nfe}, Time: {time:.2f}s, FID: {fid:.2f}")
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
