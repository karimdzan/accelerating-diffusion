from datasets import load_dataset


def download_dataset():
    # Load and cache the dataset
    print("Downloading and caching the dataset...")
    dataset = load_dataset(
        "bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images"
    )

    # Save to disk as Arrow files for reuse
    print("Saving dataset to disk...")
    dataset.save_to_disk("laion_improved_aesthetics_6.5plus_with_images")

    print(
        "Dataset successfully downloaded and saved to 'laion_improved_aesthetics_6.5plus_with_images'."
    )


if __name__ == "__main__":
    download_dataset()
