import pathlib

import datasets

datasets.disable_progress_bar()

valid_datasets = [
    "segment_image_data",
    "cluster_pixels",
    "cluster_cells",
    "post_clustering",
    "fiber_segmentation",
    "LDA_preprocessing",
    "LDA_training_inference",
    "neighborhood_analysis",
    "pairwise_spatial_enrichment",
    "ome_tiff",
]

def load_dataset(cache_dir: pathlib.Path, name: str):
    _ = datasets.load_dataset(
        path="angelolab/ark_example",
        cache_dir=cache_dir,
        name=name,
        use_auth_token=False,
        revision="main"
    )

# Make the cache directory if it doesn't exist.
cache_dir = pathlib.Path("./data/cache/")
cache_dir.mkdir(parents=True, exist_ok=True)
for dataset_config in valid_datasets:
    load_dataset(cache_dir=cache_dir.as_posix(), name=dataset_config)
