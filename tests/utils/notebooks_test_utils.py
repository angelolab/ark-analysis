import os

import numpy as np
import pandas as pd
from alpineer import image_utils


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
