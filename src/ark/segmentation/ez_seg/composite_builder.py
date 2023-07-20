import pathlib
from typing import List, Union
import numpy as np
import xarray as xr
from alpineer import misc_utils, image_utils


def composite_builder(
    data: xr.DataArray,
    images_to_add: List[str],
    images_to_subtract: List[str],
    image_type: str,
    composite_method: str,
    composite_directory: Union[str, pathlib.Path],
    composite_name: str
) -> np.ndarray:
    """
    Adds tiffs together, either pixel clusters or base signal tiffs and returns a composite channel or mask.

    Args:
        data (xr.DataArray): The data array containing the set of all images which get filtered out with images_to_add and images_to_subtract.
        images_to_add (List[str]): A list of channels or pixel cluster names to add together.
        images_to_subtract (List[str]): A list of channels or pixel cluster names to subtract from the composite.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ("binary") or intensity, gray-scale tiffs returned ("total").
        composite_directory (Union[str, pathlib.Path]): The directory to save the composite array.
        composite_name (str): The name of the composite array to save.

    Returns:
        np.ndarray: Returns the composite array, either as a binary mask, or as a scaled intensity array.
    """
    
    # TODO:
    # Switch over to spatialdata 
    image_shape = data.shape[1:]
    image_names: np.ndarray = data.fovs.values

    misc_utils.verify_in_list(images_to_add=images_to_add, image_names=image_names)
    misc_utils.verify_in_list(
        images_to_subtract=images_to_subtract, image_names=image_names
    )
    misc_utils.verify_in_list(composite_method=composite_method, options=["binary", "total"])
    
    if isinstance(composite_directory, str):
        composite_directory = pathlib.Path(composite_directory)
        composite_directory.mkdir(parents=True, exist_ok=True)
    
    composite_array = np.zeros(shape=image_shape)
    if images_to_add:
        composite_array = add_to_composite(
            data, composite_array, images_to_add, image_type, composite_method
        )
    if images_to_subtract:
        composite_array = subtract_from_composite(data, composite_array, images_to_subtract, image_type, composite_method)

    if isinstance(composite_directory, str):
        composite_directory = pathlib.Path(composite_directory)
        composite_directory.mkdir(parents=True, exist_ok=True)

    image_utils.save_image(fname=composite_directory / composite_name, data=composite_array)

    return composite_array


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
        data (xr.DataArray): The data array containing the set of all images which get filtered out with images_to_add.
        composite_array (np.ndarray): The array to add tiffs to.
        images_to_add (List[str]): A list of channels or pixel cluster names to add together.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ("binary") or intensity, gray-scale tiffs returned ("total").

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.

    """

    filtered_images: xr.DataArray = data.sel(fovs=images_to_add)

    if image_type == "signal":
        composite_array: np.ndarray = filtered_images.sum(dim="fovs").values
        if composite_method == "binary":
            composite_array = composite_array.clip(min=None, max=1)
    else:
        for fov in filtered_images.fovs.values:
            composite_array |= filtered_images.sel(fovs=fov).values
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
        data (xr.DataArray): The data array containing the set of all images which get filtered out with images_to_subtract.
        composite_array (np.ndarray): An array to subtract tiffs from.
        images_to_subtract (List[str]): A list of channels or pixel cluster names to subtract from the composite.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ('binary') or intensity, gray-scale tiffs returned ('total').

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.
    """

    filtered_images: xr.DataArray = data.sel(fovs=images_to_subtract)
    # for each channel to subtract
    for channel in filtered_images.fovs.values:
        channel_data = filtered_images.sel(fovs=channel)
        # if signal-based data
        if image_type == "signal":
            # if a binarized, or binary removal is asked for
            if composite_method == "binary":
                # Create a mask based on positive values in the subtraction channel
                mask_2_zero = channel_data > 0
                # Zero out elements in the composite channel based on mask
                composite_array[mask_2_zero] = 0
                # return binarized composite
                composite_array[composite_array > 1] = 1
            # if a signal based, or partial, removal is asked for
            elif composite_method == "total":
                # subtract channel counts from composite array
                composite_array = np.subtract(composite_array, channel_data)
                # Find the minimum value in the composite array
                min_value = composite_array.min()
                # If the minimum value is negative, add its absolute value to all elements
                if min_value < 0:
                    composite_array += abs(min_value)
        # else if pixel clustered data
        elif image_type == "pixel_clustered":
            # subtract positive pixels to composite array
            composite_array -= channel_data
            # zero out any negative values
            composite_array.clip(min=0, max=None)
    # return the composite array
    return composite_array
