import modal

app = modal.App("nanochat-dataset-downloader")

volume = modal.Volume.from_name("nanochat", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("datasets", "huggingface_hub")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def download_dataset():
    from datasets import load_dataset

    dataset_kwargs = {
        "path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "name": "sample-10BT",
    }

    print("Downloading dataset...")
    ds = load_dataset(**dataset_kwargs)

    print(f"Dataset downloaded. Total rows: {len(ds)}")

    cache_dir = "/data/fineweb-edu"
    print(f"Saving dataset to {cache_dir}...")
    ds.save_to_disk(cache_dir)

    volume.commit()
    print("Dataset saved to Modal volume 'nanochat'")

    return {"status": "success", "rows": len(ds), "cache_dir": cache_dir}

@app.local_entrypoint()
def main():
    result = download_dataset.remote()
    print(f"Download complete: {result}")
