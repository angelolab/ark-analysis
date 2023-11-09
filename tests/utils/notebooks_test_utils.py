from functools import wraps
import os
from typing import Union
import shutil
import numpy as np
import pandas as pd
from alpineer import image_utils

from ark.settings import EXAMPLE_DATASET_REVISION
from ark.utils import example_dataset


def create_pixel_remap_files(base_dir,  pixel_meta_cluster_mapping):
    """
    Generates the pixel_remap file for simulating the metaclustering gui.

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        pixel_meta_cluster_mapping (str): The file location to save the remapped metaclusters to.
    """
    # define the remapping file
    remap_data = pd.DataFrame(
        np.random.rand(100, 3),
        columns=['pixel_som_cluster', 'pixel_meta_cluster', 'pixel_meta_cluster_rename']
    )
    remap_data['pixel_som_cluster'] = range(1, 101)
    remap_data['pixel_meta_cluster'] = np.repeat(range(1, 11), 10)
    remap_data['pixel_meta_cluster_rename'] = np.repeat(
        ['meta_' + str(i) for i in range(1, 11)], 10
    )
    remap_data.to_csv(os.path.join(base_dir, pixel_meta_cluster_mapping), index=False)


def create_cell_remap_files(base_dir,  cell_meta_cluster_remap):
    """
    Generates the cell_remap file for simulating the metaclustering gui.

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        cell_meta_cluster_remap (str): The file location to save the remapped metaclusters to.
    """
    # define the remapping file
    remap_data = pd.DataFrame(
        np.random.rand(100, 3),
        columns=['cell_som_cluster', 'cell_meta_cluster', 'cell_meta_cluster_rename']
    )
    remap_data['cell_som_cluster'] = range(1, 101)
    remap_data['cell_meta_cluster'] = np.repeat(range(1, 11), 10)
    remap_data['cell_meta_cluster_rename'] = np.repeat(
        ['meta_' + str(i) for i in range(1, 11)], 10
    )
    remap_data.to_csv(os.path.join(base_dir, cell_meta_cluster_remap), index=False)


def generate_sample_feature_tifs(fovs, deepcell_output_dir, img_shape=(50, 50)):
    """Generate a sample _whole_cell and _nuclear tiff file for each fov

    Done to bypass the bottleneck of create_deepcell_output, for testing purposes we don't care
    about correct segmentation labels.

    Args:
        fovs (list): The list of fovs to generate sample tiff files for
        deepcell_output_dir (str): The path to the output directory
        img_shape (tuple): Dimensions of the tifs to create
    """

    # generate a random image for each fov, set as both whole cell and nuclear
    for fov in fovs:
        rand_img = np.random.randint(0, 16, size=img_shape)
        image_utils.save_image(os.path.join(deepcell_output_dir, fov + "_whole_cell.tiff"),
                               rand_img)
        image_utils.save_image(os.path.join(deepcell_output_dir, fov + "_nuclear.tiff"),
                               rand_img)


def _ex_dataset_download(dataset: str, save_dir: str, cache_dir: Union[str, None]):
    """Downloads the example dataset and moves it to the save_dir.

    Args:
        dataset (str): The name of the dataset to download.
        save_dir (str): The directory to save the dataset to.
        cache_dir (Union[str, None]): The directory to cache the dataset to.
    """
    overwrite_existing = True
    ex_dataset = example_dataset.ExampleDataset(dataset=dataset,
                                                overwrite_existing=overwrite_existing,
                                                cache_dir=cache_dir,
                                                revision=EXAMPLE_DATASET_REVISION)
    ex_dataset.download_example_dataset()

    ex_dataset.move_example_dataset(move_dir=save_dir)


def get_storage(method):
    """
    Gets the storage after the method is run. Used for debugging storage isses
    on CI in the notebooks.

    Args:
        method (Callable): The function must be a class method, with access to `self` and
        `base_dir`.

    Returns:
        method: Returns the method
    """
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)

        total, used, free = shutil.disk_usage(self.base_dir)
        print(f"After: {method.__name__}")
        print(f"Total: {total // (2**20)} MiB")
        print(f"Used: {used // (2**20)} MiB")
        print(f"Free: {free // (2**20)} MiB")
        return method_output
    return _impl
