import pathlib
from typing import Optional, Tuple, Union, Literal
import numpy as np
from skimage import measure, filters, morphology
from skimage.util import map_array
import pandas as pd
from alpineer import misc_utils, load_utils, image_utils, io_utils
from ark.segmentation.ez_seg.ez_seg_utils import log_creator
import xarray as xr
import warnings


def create_object_masks(
        image_data_dir: Union[str, pathlib.Path],
        img_sub_folder: Optional[str],
        fov_list: list[str],
        mask_name: str,
        channel_to_segment: str,
        masks_dir: Union[str, pathlib.Path],
        log_dir: Union[str, pathlib.Path],
        object_shape_type: str = "blob",
        sigma: int = 1,
        thresh: Optional[np.float32] = None,
        hole_size: Optional[int] = None,
        fov_dim: int = 400,
        min_object_area: int = 100,
        max_object_area: int = 100000,
) -> None:
    """
    Calculates a mask for each channel in the FOV for circular or 'blob'-like objects such as: single large cells or amyloid
    plaques. It will blur the input image, then threshold the blurred image on either a given
    fixed value, or an adaptive thresholding method. In addition, it removes small holes using
    that same thresholding input and filters out objects which are either too small or too large.

    Args:
        image_data_dir (Union[str, pathlib.Path]): The directory to pull images from to perform segmentation on.
        img_sub_folder (str): A name for sub-folders within each fov in the image_data location.
        fov_list: A list of fov names to segment on.
        mask_name (str): The name of the masks you are creating.
        channel_to_segment: The channel on which to perform segmentation.
        masks_dir (Union[str, pathlib.Path]): The directory to save segmented images to.
        object_shape_type (str, optional): Specify whether the object is either "blob" or
        "projection" shaped. Defaults to "blob".
        sigma (int): The standard deviation for Gaussian kernel, used for blurring the
        image. Defaults to 1.
        thresh (np.float32, optional): The global threshold value for image thresholding if
        desired. Defaults to None.
        hole_size (int, optional): A specific area to close small holes over in object masks.
        Defaults to None.
        fov_dim (int): The dimension in μm of the FOV.
        min_object_area (int): The minimum size (area) of an object to capture in
        pixels. Defaults to 100.
        max_object_area (int): The maximum size (area) of an object to capture in
        pixels. Defaults to 100000.
        log_dir (Union[str, pathlib.Path]): The directory to save log information to.
    """

    # Input validation
    io_utils.validate_paths([image_data_dir, masks_dir, log_dir])

    misc_utils.verify_in_list(
        object_shape=[object_shape_type], object_shape_options=["blob", "projection"]
    )

    for fov in fov_list:
        fov_xr: xr.DataArray = load_utils.load_imgs_from_tree(
            data_dir=image_data_dir, img_sub_folder=img_sub_folder, fovs=fov
        ).squeeze()

        # handles folders where only 1 channel is loaded, often for composites
        try:
            len(fov_xr.channels)
            channel: xr.DataArray = fov_xr.sel({"channels": channel_to_segment}).astype(
                np.float32
            )
        except TypeError:
            channel: xr.DataArray = fov_xr.astype(np.float32)

        object_masks: np.ndarray = _create_object_mask(
            input_image=channel,
            object_shape_type=object_shape_type,
            sigma=sigma,
            thresh=thresh,
            hole_size=hole_size,
            fov_dim=fov_dim,
            min_object_area=min_object_area,
            max_object_area=max_object_area,
        )

        # save the channel overlay
        save_path = pathlib.Path(masks_dir) / f"{fov}_{mask_name}.tiff"
        image_utils.save_image(fname=save_path, data=object_masks)

    # Write a log saving ez segment info
    variables_to_log = {
        "image_data_dir": image_data_dir,
        "fov_list": fov_list,
        "mask_name": mask_name,
        "channel_to_segment": channel_to_segment,
        "masks_dir": masks_dir,
        "object_shape_type": object_shape_type,
        "sigma": sigma,
        "thresh": thresh,
        "hole_size": hole_size,
        "fov_dim": fov_dim,
        "min_object_area": min_object_area,
        "max_object_area": max_object_area,
    }
    log_creator(variables_to_log, log_dir, f"{mask_name}_segmentation_log.txt")
    print("ez masks built and saved")


def _create_object_mask(
        input_image: xr.DataArray,
        object_shape_type: Union[Literal["blob"], Literal["projection"]] = "blob",
        sigma: int = 1,
        thresh: Union[int, Literal["auto"]] = None,
        hole_size: Union[int, Literal["auto"]] = "auto",
        fov_dim: int = 400,
        min_object_area: int = 10,
        max_object_area: int = 100000,
) -> np.ndarray:
    """
    Calculates a mask for circular or 'blob'-like objects such as: single large cells or amyloid
    plaques. It will blur the input image, then threshold the blurred image on either a given
    fixed value, or an adaptive thresholding method. In addition, it removes small holes using
    that same thresholding input and filters out objects which are either too small or too large.

    Args:
        input_image (xr.DataArray): The numpy array (image) to perform segmentation on.
        object_shape_type (str, optional): Specify whether the object is either "blob" or
        "projection" shaped. Defaults to "blob".
        sigma (int): The standard deviation for Gaussian kernel, used for blurring the
        image. Defaults to 1.
        thresh (int, str,  optional): The global threshold value for image thresholding if
        desired. Defaults to "auto".
        hole_size (int, str, optional): A specific area to close small holes over in object masks.
        Defaults to None.
        fov_dim (int): The dimension in μm of the FOV.
        min_object_area (int): The minimum size (area) of an object to capture in
        pixels. Defaults to 100.
        max_object_area (int): The maximum size (area) of an object to capture in
        pixels. Defaults to 100000.

    Returns:
        np.ndarray: The object mask.
    """

    # Do not display any UserWarning msg's about boolean arrays here.
    warnings.filterwarnings(
        "ignore", message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?")

    # Input validation
    misc_utils.verify_in_list(object_shape_type=[object_shape_type], object_shape_options=["blob", "projection"])

    # Copy the input image, and get its shape
    img2mask: np.ndarray = input_image.copy()
    if type(input_image) != np.ndarray:
        img2mask = img2mask.to_numpy()
    img2mask_shape: Tuple[int, int] = img2mask.shape

    # Blur the input mask using given sigma value
    if sigma is None:
        img2mask_blur = img2mask
    else:
        img2mask_blur: np.ndarray = filters.gaussian(
            img2mask, sigma=sigma, preserve_range=True
        )

    # Apply binary thresholding to the blurred image
    if isinstance(thresh, int):
        # Find the threshold value based on the given percentile number
        img_nonzero = img2mask_blur[img2mask_blur != 0]
        thresh_percentile = np.percentile(img_nonzero, thresh)

        # Threshold the array where values below the threshold are set to 0
        img2mask_thresh = np.where(img2mask_blur < thresh_percentile, 0, img2mask_blur)

    elif thresh == "auto":
        local_thresh_block_size: int = get_block_size(
            block_type="local_thresh", fov_dim=fov_dim, img_shape=img2mask_shape[0]
        )
        img2mask_thresh: np.ndarray = img2mask_blur > filters.threshold_local(
            img2mask_blur, block_size=local_thresh_block_size
        )
    elif thresh is None:
        img2mask_thresh = img2mask_blur
    else:
        raise ValueError(
            f"Invalid `threshold` value: {thresh}. Must be either `auto`, `None` or an integer."
        )

    # Binarize the image in prep for removing holes.
    img2mask_thresh_binary = img2mask_thresh > 0
    img2mask_thresh[img2mask_thresh_binary] = 1
    img2mask_thresh = img2mask_thresh.astype(int)

    # Remove small holes within the objects
    if isinstance(hole_size, int):
        img2mask_rm_holes: np.ndarray = morphology.remove_small_holes(
            img2mask_thresh, area_threshold=hole_size
        )
    elif hole_size == "auto":
        small_holes_block_size: int = get_block_size(
            block_type="small_holes", fov_dim=fov_dim, img_shape=img2mask_shape[0]
        )
        img2mask_rm_holes: np.ndarray = morphology.remove_small_holes(
            img2mask_thresh, area_threshold=small_holes_block_size
        )
    elif hole_size is None:
        img2mask_rm_holes = img2mask_thresh
    else:
        raise ValueError(
            f"Invalid `hole_size` value: {hole_size}. Must be either `auto`, `None` or an integer."
        )

    # Filter projections
    if object_shape_type == "projection":
        img2mask_filtered: np.ndarray = filters.meijering(
            img2mask_rm_holes, sigmas=range(1, 5, 1), black_ridges=False
        )
    else:
        img2mask_filtered: np.ndarray = img2mask_rm_holes

    # Binarize the image in prep for labeling.
    img2mask_filtered_binary = img2mask_filtered > 0
    img2mask_filtered[img2mask_filtered_binary] = 1

    # Extract `label` and `area` from regionprops
    labeled_object_masks = measure.label(img2mask_filtered, connectivity=2)

    # Convert dictionary of region properties to DataFrame
    object_masks_df: pd.DataFrame = pd.DataFrame(
        measure.regionprops_table(
            label_image=labeled_object_masks,
            cache=True,
            properties=[
                "label",
                "area",
            ],
        )
    )

    # zero out objects not meeting size requirements
    keep_labels_bool = (object_masks_df["area"] >= min_object_area) & (
            object_masks_df["area"] <= max_object_area
    )
    all_labels = object_masks_df["label"]
    labels_to_keep = all_labels * keep_labels_bool

    # Map filtered objects to the object mask
    objects_filtered_by_size = map_array(
        labeled_object_masks, all_labels.to_numpy(), labels_to_keep.to_numpy()
    )

    return objects_filtered_by_size.astype(np.int32)


def get_block_size(block_type: str, fov_dim: int, img_shape: int) -> int:
    """
    Computes the approximate local otsu threshold based on fov size (in μm) and pixel resolution.

    Args:
        block_type (str): Either "small_holes" or "local_thresh"
        fov_dim (int, optional): The size in μm for the FOV.
        img_shape (int, optional): The shape of the image.

    Returns:
        int: Returns the approximate block area
    """

    # Input validation
    misc_utils.verify_in_list(
        block_type=[block_type], block_types=["small_holes", "local_thresh"]
    )
    # Get the size of the pixel

    pixel_size = fov_dim / img_shape
    # grab block size for removing small holes
    if block_type == "small_holes":
        size = (np.pi * 5) ** 2 / pixel_size
        # round up above value to make it into an integer
        area: int = round(size)
    # grab local threshold block size
    else:
        # use this to calculate out how many pixels it takes to get to roughly 10 μm
        # (roughly a cell soma diameter)
        size: float = 10 / pixel_size

        # round the area up to the nearest odd number
        area: int = round(size)
        if area % 2 == 0:
            area += 1
    return area
