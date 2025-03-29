from pathlib import Path
import os
from ark.utils.example_dataset import get_example_dataset, DatasetConfig

repo_root = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()


# Create the save directory
save_dir = repo_root / "data"
save_dir.mkdir(parents=True, exist_ok=True)

# Create a cache directory 
cache_dir = save_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Download all available datasets using our implementation
for dataset_config in DatasetConfig:
    print(f"Downloading dataset: {dataset_config.value}")
    get_example_dataset(
        dataset=dataset_config.value,
        save_dir=save_dir, 
        overwrite_existing=True,
        revision="main"
    )

print("All datasets downloaded successfully to:", save_dir)
