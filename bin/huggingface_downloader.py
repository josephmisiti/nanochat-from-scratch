from datasets import load_dataset

dataset_kwargs = {
    "path": "HuggingFaceFW/fineweb-edu",
    "split": "train",
    "name": "sample-10BT", # ~100B GPT-2 tokens at ~3 chars/token => ~300B chars total
}
ds = load_dataset(**dataset_kwargs)

