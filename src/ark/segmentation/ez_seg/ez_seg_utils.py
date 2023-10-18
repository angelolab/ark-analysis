from typing import Generator, Union
from skimage.io import imread
from alpineer.image_utils import save_image
from alpineer import io_utils
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
import pathlib


def renumber_masks(
        mask_dir: Union[pathlib.Path, str]
):
    """
    Relabels all masks in mask tiffs so each label is unique across all mask images in entire dataset.
    Args:
        mask_dir (Union[pathlib.Path, str]): Directory that points to parent directory of all segmentation masks to be relabeled.
    """
    mask_dir = pathlib.Path(mask_dir)
    io_utils.validate_paths(mask_dir)

    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    global_unique_labels = 1

    # First pass - get total number of unique masks
    for image in all_images:
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                global_unique_labels += 1

    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    # Second pass - relabel all masks starting at unique num of masks +1
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
        mantis_dir: Union[str, pathlib.Path],
) -> None:
    """
    Creates a folder for viewing FOVs in Mantis.

    Args:
        fovs (str | list[str]):
            A list of FOVs to use for creating the mantis project
        tiff_dir (Union[str, pathlib.Path]):
            The path to the directory containing the raw image data.
        mask_dir (Union[str, pathlib.Path]):
            The path to the directory containing the masks.
        mantis_dir:
            The path to the directory containing housing the ez_seg specific mantis project.
    """
    for fov in tqdm(io_utils.list_folders(tiff_dir, substrs=fovs)):
        shutil.copytree(os.path.join(tiff_dir, fov), dst=os.path.join(mantis_dir, fov))

        for mask in io_utils.list_files(mask_dir, substrs=fov):
            shutil.copy(os.path.join(mask_dir, mask), os.path.join(mantis_dir, fov))


def log_creator(variables_to_log: dict, base_dir: str, log_name: str = "config_values.txt"):
    # Define the filename for the text file
    output_file = os.path.join(base_dir, log_name)

    # Open the file in write mode and write the variable values
    with open(output_file, "w") as file:
        for variable_name, variable_value in variables_to_log.items():
            file.write(f"{variable_name}: {variable_value}\n")

    print(f"Values saved to {output_file}")
