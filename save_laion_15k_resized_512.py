from datasets import load_from_disk, Dataset
from PIL import Image
import io
import random
from tqdm import tqdm
import gc

dataset_path = "laion_improved_aesthetics_6.5plus_with_images"
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]

sample_size = 15000
random_indices = random.sample(range(len(train_dataset)), sample_size)
sampled_dataset = train_dataset.select(random_indices)

resized_samples = []

for sample in tqdm(sampled_dataset):
    image_bytes = sample["image"]["bytes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    resized_image = image.resize((512, 512), Image.LANCZOS)

    buffer = io.BytesIO()
    resized_image.save(buffer, format="JPEG")
    buffer.seek(0)

    resized_samples.append({"image": {"bytes": buffer.read()}, "text": sample["text"]})

    buffer.close()
    del buffer

gc.collect()

resized_dataset = Dataset.from_list(resized_samples)

output_path = "./resized_laion_512x512"
resized_dataset.save_to_disk(output_path)

print("Saved")
