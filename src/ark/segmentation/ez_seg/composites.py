import pathlib
from typing import List, Union
import numpy as np
import xarray as xr
from alpineer import misc_utils, image_utils, load_utils
from ark.segmentation.ez_seg.ez_seg_utils import log_creator


def composite_builder(
    image_data_dir: Union[str, pathlib.Path],
    img_sub_folder: str,
    fov_list: list[str],
    images_to_add: list[str],
    images_to_subtract: list[str],
    image_type: str,
    composite_method: str,
    composite_directory: Union[str, pathlib.Path] = None,
    composite_name: str = None,
    log_dir: Union[str, pathlib.Path] = None,
) -> None:
    """
    Adds tiffs together, either pixel clusters or base signal tiffs and returns a composite channel or mask.

    Args:
        image_data_dir (Union[str, pathlib.Path]): The path to dir containing the set of all images 
            which get filtered out with `images_to_add` and `images_to_subtract`.
        img_sub_folder (str): A name for sub-folders within each fov in the image_data location.
        fov_list: A list of fov's to create composite channels through.
        images_to_add (List[str]): A list of channels or pixel cluster names to add together.
        images_to_subtract (List[str]): A list of channels or pixel cluster names to subtract 
            from the composite.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ("binary") or intensity, gray-scale tiffs 
            returned ("total").
        composite_directory (Union[str, pathlib.Path]): The directory to save the composite array.
        composite_name (str): The name of the composite array to save.
        log_dir: The directory to save log information to.

    Returns:
        np.ndarray: Saves the composite array, either as a binary mask, or as a scaled intensity array.
        If composite_directory is None, return a dictionary with keys being FOV names and values
        are np.ndarray of the composite image.
    """
    composite_images = {}
    for fov in fov_list:
        # load in tiff images and verify channels are present
        fov_data = load_utils.load_imgs_from_tree(
            data_dir=image_data_dir, img_sub_folder=img_sub_folder, fovs=fov
        )

        image_shape = fov_data.shape[1:3]

        misc_utils.verify_in_list(
            images_to_add=images_to_add, image_names=fov_data.channels.values
        )
        misc_utils.verify_in_list(
            images_to_subtract=images_to_subtract, image_names=fov_data.channels.values
        )
        misc_utils.verify_in_list(
            composite_method=composite_method, options=["binary", "total"]
        )

        # Initialize composite array, and add & subtract channels
        composite_array = np.zeros(shape=image_shape, dtype=np.float32)
        if images_to_add:
            composite_array = add_to_composite(
                fov_data, composite_array, images_to_add, image_type, composite_method
            )
        if images_to_subtract:
            composite_array = subtract_from_composite(
                fov_data, composite_array, images_to_subtract, image_type, composite_method
            )

        if composite_directory:
            # Create the fov dir within the composite dir
            composite_fov_dir = pathlib.Path(composite_directory) / fov
            composite_fov_dir.mkdir(parents=True, exist_ok=True)

            # Save the composite image
            image_utils.save_image(
                fname=pathlib.Path(composite_directory) / fov / f"{composite_name}.tiff",
                data=composite_array.astype(np.uint32)
            )

        composite_images[fov] = composite_array.astype(np.float32)
        
    # Write a log saving composite builder info
    if log_dir:
        variables_to_log = {
            "image_data_dir": image_data_dir,
            "fov_list": fov_list,
            "images_to_add": images_to_add,
            "images_to_subtract": images_to_subtract,
            "image_type": image_type,
            "composite_method": composite_method,
            "composite_directory": composite_directory,
            "composite_name": composite_name,
        }
        log_creator(variables_to_log, log_dir, f"{composite_name}_composite_log.txt")
    else:
        return composite_images

    print("Composites built and saved")


def add_to_composite(
    data: xr.DataArray,
    composite_array: np.ndarray,
    images_to_add: List[str],
    image_type: str,
    composite_method: str,
) -> np.ndarray:
    """
    Adds tiffs together to form a composite array.

    Args:
        data (xr.DataArray): The data array containing the set of all images which get filtered out
            with `images_to_add`.
        composite_array (np.ndarray): The array to add channels to.
        images_to_add (List[str]): A list of channels or pixel cluster names to add together.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ("binary") or intensity, gray-scale tiffs
            returned ("total").

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.
    """

    filtered_channels: xr.DataArray = data.sel(
        {"channels": images_to_add}).squeeze().astype(np.float32)
    if len(images_to_add) > 1:
        composite_array = filtered_channels.sum(dim="channels").values
    else:
        composite_array = filtered_channels
    if image_type == "pixel_cluster" or composite_method == "binary":
        composite_array = composite_array.clip(min=None, max=1)

    return composite_array


def subtract_from_composite(
    data: xr.DataArray,
    composite_array: np.ndarray,
    images_to_subtract: List[str],
    image_type: str,
    composite_method: str,
) -> np.ndarray:
    """
    Subtracts tiffs from a composite array.

    Args:
        data (xr.DataArray): The data array containing the set of all images which get
            filtered out with `images_to_subtract`.
        composite_array (np.ndarray): An array to subtract channels from.
        images_to_subtract (List[str]): A list of channels or pixel cluster names to subtract
            from the composite.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ('binary') or intensity, gray-scale tiffs
            returned ('total').

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.
    """

    filtered_channels: xr.DataArray = data.sel(
        {"channels": images_to_subtract}).squeeze().astype(np.float32)
    if len(images_to_subtract) > 1:
        composite_array2sub = filtered_channels.sum(dim="channels").values
    else:
        composite_array2sub = filtered_channels

    if image_type == "signal" and composite_method == "binary":
        mask_2_zero = composite_array2sub > 0
        composite_array[mask_2_zero] = 0
        composite_array[composite_array > 1] = 1

    else:
        composite_array -= composite_array2sub
        composite_array = composite_array.clip(min=0, max=None)

    return composite_array
