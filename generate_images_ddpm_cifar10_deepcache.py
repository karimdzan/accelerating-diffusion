import torch
from diffusers import DDIMPipeline
from torchvision.utils import save_image
from torchvision.transforms import ToTensor

import os
from DeepCache import DeepCacheSDHelper
from tqdm import tqdm

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-cifar10-32"
ddpm_pipeline = DDIMPipeline.from_pretrained(model_id).to(device)
ddpm_pipeline.set_progress_bar_config(disable=True)

# helper = DeepCacheSDHelper(pipe=ddpm_pipeline)
# helper.set_params(cache_interval=10, cache_branch_id=0, skip_mode="uniform")
# helper.enable()

# Global index for unique filenames
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

global_index = 0
total_images = 10000
batch_size = 64
num_batches = total_images // batch_size
remaining_images = total_images % batch_size

torch.cuda.synchronize()  # Ensure all previous GPU operations are complete
start_event.record()
# Generate images in batches
for j in tqdm(range(num_batches)):
    with torch.no_grad():
        images = ddpm_pipeline(batch_size=batch_size).images

    for i, img in enumerate(images):
        save_image(ToTensor()(img), os.path.join(output_dir, f"img_{global_index}.png"))
        global_index += 1  # Increment global index

# Generate remaining images
if remaining_images > 0:
    with torch.no_grad():
        images = ddpm_pipeline(batch_size=remaining_images).images

    for i, img in enumerate(images):
        save_image(ToTensor()(img), os.path.join(output_dir, f"img_{global_index}.png"))
        global_index += 1  # Increment global index

end_event.record()
torch.cuda.synchronize()  # Ensure all GPU operations are complete

elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds

throughput = total_images / (elapsed_time / 1000)

print(f"Generated {global_index} images.")
print(f"Total time with DeepCache: {elapsed_time / 1000:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
