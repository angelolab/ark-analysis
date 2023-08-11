import pathlib
from typing import List, Union
import numpy as np
import xarray as xr
from alpineer import misc_utils, image_utils, load_utils


def composite_builder(
    data_dir: Union[str, pathlib.Path],
    fov: str,
    images_to_add: list[str],
    images_to_subtract: list[str],
    image_type: str,
    composite_method: str,
    composite_directory: Union[str, pathlib.Path],
    composite_name: str,
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
    
    fov_data = load_utils.load_imgs_from_tree(data_dir=data_dir, fovs=fov)


    image_shape = fov_data.shape[1:]

    misc_utils.verify_in_list(
        images_to_add=images_to_add, image_names=fov_data.channels.values
    )
    misc_utils.verify_in_list(
        images_to_subtract=images_to_subtract, image_names=fov_data.channels.values
    )
    misc_utils.verify_in_list(
        composite_method=composite_method, options=["binary", "total"]
    )

    if isinstance(composite_directory, str):
        composite_directory = pathlib.Path(composite_directory)
        composite_directory.mkdir(parents=True, exist_ok=True)

    composite_array = np.zeros(shape=image_shape)
    if images_to_add:
        composite_array = add_to_composite(
            fov_data, composite_array, images_to_add, image_type, composite_method
        )
    if images_to_subtract:
        composite_array = subtract_from_composite(
            fov_data, composite_array, images_to_subtract, image_type, composite_method
        )

    if isinstance(composite_directory, str):
        composite_directory = pathlib.Path(composite_directory)
        composite_directory.mkdir(parents=True, exist_ok=True)
    
    composite_fov_dir = composite_directory / fov
    composite_fov_dir.mkdir(parents=True, exist_ok=True)

    image_utils.save_image(
        fname=composite_directory / fov / (f"{composite_name}.tiff"), data=composite_array
    )


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
        composite_array (np.ndarray): The array to add channels to.
        images_to_add (List[str]): A list of channels or pixel cluster names to add together.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ("binary") or intensity, gray-scale tiffs returned ("total").

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.

    """

    filtered_channels: xr.DataArray = data.sel({"channels": images_to_add}).squeeze()

    if image_type == "signal":
        composite_array: np.ndarray = filtered_channels.sum(dim="channels").values
        if composite_method == "binary":
            composite_array = composite_array.clip(min=None, max=1)
    else:
        for fov in filtered_channels.fovs.values:
            composite_array |= filtered_channels.sel(fovs=fov).values
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
        composite_array (np.ndarray): An array to subtract channels from.
        images_to_subtract (List[str]): A list of channels or pixel cluster names to subtract from the composite.
        image_type (str): Either "signal" or "pixel_cluster" data.
        composite_method (str): Binarized mask returns ('binary') or intensity, gray-scale tiffs returned ('total').

    Returns:
        np.ndarray: The composite array, either as a binary mask, or as a scaled intensity array.
    """

    filtered_channels: xr.DataArray = data.sel(
        {"channels": images_to_subtract}
    ).squeeze()
    # for each channel to subtract
    for channel in filtered_channels.channels.values:
        channel_data = filtered_channels.sel(channels=channel)
        if image_type == "signal" and composite_method == "binary":
            mask_2_zero = channel_data > 0
            composite_array[mask_2_zero] = 0
            composite_array[composite_array > 1] = 1

        elif image_type == "signal" and composite_method == "total":
            composite_array -= channel_data
            min_value = composite_array.min()
            if min_value < 0:
                composite_array -= min_value
        elif image_type == "pixel_cluster":
            composite_array -= channel_data
            composite_array.clip(min=0, max=None)
    return composite_array
