import torch
import torchvision.transforms as T
from datasets import load_dataset
from piq import FID
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from diffusers import ConsistencyModelPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = ConsistencyModelPipeline.from_pretrained(
    "openai/diffusers-cd_imagenet64_lpips", torch_dtype=torch.float16
).to(device)
pipeline.set_progress_bar_config(disable=True)

dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", split="test")

transform = T.Compose(
    [
        T.ToTensor(),
    ]
)

fid = FID()


class SingleImageDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return {"images": self.tensor[idx]}


gen_tensor = []
img_tensor = []

# selected_data_test = dataset.select(range(2))

for i in tqdm(range(0, len(dataset), 64)):
    with torch.no_grad():
        generated_img = pipeline(num_inference_steps=2, batch_size=64)

        gen_pil = generated_img.images
    batch = dataset[i : i + 64]["image"]

    transformed_batch = torch.stack([transform(img) for img in batch])
    transformed_gen_batch = torch.stack([transform(img) for img in gen_pil])

    img_tensor.append(transformed_batch)
    gen_tensor.append(transformed_gen_batch)

img_tensor = torch.cat(img_tensor, dim=0)
gen_tensor = torch.cat(gen_tensor, dim=0)

img_loader = DataLoader(SingleImageDataset(img_tensor), batch_size=64)
gen_loader = DataLoader(SingleImageDataset(gen_tensor), batch_size=64)

img_feats = fid.compute_feats(img_loader)
gen_feats = fid.compute_feats(gen_loader)

fid_score = fid(img_feats, gen_feats)

print(fid_score)
