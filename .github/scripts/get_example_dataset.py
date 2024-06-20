from pathlib import Path
import os

import datasets

datasets.disable_progress_bar()

DATASET_PATH = "angelolab/ark_example"

valid_configs = datasets.get_dataset_config_names(DATASET_PATH, trust_remote_code=True)

def load_dataset(cache_dir: Path, name: str):
    _ = datasets.load_dataset(
        path=DATASET_PATH,
        cache_dir=cache_dir,
        name=name,
        token=False,
        revision="main",
        trust_remote_code=True,
    )

# Create the cache directory
cache_dir = Path(os.environ.get("GITHUB_WORKSPACE")).resolve() / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Download all available datasets
for dataset_config in valid_configs:
    load_dataset(cache_dir=cache_dir, name=dataset_config)
