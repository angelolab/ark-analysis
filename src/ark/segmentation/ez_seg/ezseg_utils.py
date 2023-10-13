from typing import Generator, Union
from skimage.io import imread
from alpineer.image_utils import save_image
from alpineer import io_utils
import os, shutil
from tqdm.auto import tqdm
import numpy as np
import pathlib

def renumber_masks(mask_dir):
    mask_dir = pathlib.Path(mask_dir)
    io_utils.validate_paths(mask_dir)
    
    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    global_unique_labels = 1

    for image in all_images:
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                global_unique_labels += 1

    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    for image in all_images:
        print("Relabeling: " + image.stem + image.suffix)
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                img[img == label] = global_unique_labels
                global_unique_labels += 1
        save_image(fname=image, data=img)
    print("Complete.")




def create_mantis_project(
        fovs: str | list[str],
        tiff_dir: Union[str, pathlib.Path],
        mask_dir: Union[str, pathlib.Path],
        segmentation_dir: Union[str, pathlib.Path],
        ez_visualization_dir: Union[str, pathlib.Path],
):
    """

    Args:
        fovs (str | list[str]):
            A list of FOVs to use for creating the mantis project
        tiff_dir (Union[str, pathlib.Path]):
            The path to the directory containing the raw image data.
        mask_dir (Union[str, pathlib.Path]):
            The path to the directory containing the masks.
        segmentation_dir (Union[str, pathlib.Path]):
            The path to the directory containing the segmentation data.
        ez_visualization_dir:
            The path to the directory containing housing the ezseg specific mantis project.

    Returns:

    """
    for fov in tqdm(io_utils.list_folders(tiff_dir, substrs=fovs)):
        shutil.copytree(os.path.join(tiff_dir, fov), dst=os.path.join(ez_visualization_dir, fov))

        for mask in io_utils.list_files(mask_dir, substrs=fov):
            shutil.copy(os.path.join(mask_dir, mask), os.path.join(ez_visualization_dir, fov))

        for sg_mask in io_utils.list_files(os.path.join(segmentation_dir, "deepcell_output"), substrs=fov):
            shutil.copy(os.path.join(segmentation_dir, "deepcell_output", sg_mask), os.path.join(ez_visualization_dir, fov))