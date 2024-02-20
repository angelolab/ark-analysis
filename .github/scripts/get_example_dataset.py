import pathlib

import datasets

datasets.disable_progress_bar()

DATASET_PATH = "angelolab/ark_example"

valid_configs = datasets.get_dataset_config_names(DATASET_PATH)

def load_dataset(cache_dir: pathlib.Path, name: str):
    _ = datasets.load_dataset(
        path=DATASET_PATH,
        cache_dir=cache_dir,
        name=name,
        token=False,
        revision="main",
        trust_remote_code=True,
    )

# Make the cache directory if it doesn't exist.
cache_dir = pathlib.Path("~/.cache/huggingface/datasets")
cache_dir.mkdir(parents=True, exist_ok=True)
for dataset_config in valid_configs:
    load_dataset(cache_dir=cache_dir.as_posix(), name=dataset_config)
