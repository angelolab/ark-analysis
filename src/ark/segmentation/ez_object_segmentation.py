from typing import Dict, Optional, Tuple
import numpy as np
from skimage import measure, filters, morphology
from skimage.util import map_array
import pandas as pd
from alpineer import misc_utils


def create_object_masks(
    input_image: np.ndarray,
    object_shape_type: str = "blob",
    sigma: int = 1,
    thresh: Optional[np.float32] = None,
    hole_size: Optional[int] = None,
    fov_dim: int = 400,
    min_object_area: int = 100,
    max_object_area: int = 100000,
) -> np.ndarray:
    """
    Calculates a mask for circular or 'blob'-like objects such as: single large cells or amyloid
    plaques. It will blur the input image, then threshold the blurred image on either a given
    fixed value, or an adaptive thresholding method. In addition it removes small holes using
    that same thresholding input and filters out objects which are either too small or too large.

    Args:
        input_image (np.ndarray): The numpy array (image) to perform segmentation on.
        object_shape_type (str, optional): Specify whether the object is either "blob" or
        "projection" shaped. Defaults to "blob".
        sigma (int): The standard deviation for Gaussian kernel, used for bluring the
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

    Returns:
        np.ndarray: The object mask.
    """

    # Input validation
    misc_utils.verify_in_list(
        object_shape=[object_shape_type], object_shape_options=["blob", "projection"]
    )

    # Copy the input image, and get it's shape
    img2mask: np.ndarray = input_image.copy()
    img2mask_shape: Tuple[int, int] = img2mask.shape

    # Blur the input mask using given sigma value
    img2mask_blur: np.ndarray = filters.gaussian(img2mask, sigma=sigma)

    # Apply binary thresholding to the blurred image
    if thresh is not None:
        img2mask_thresh = img2mask_blur > thresh
    else:
        local_thresh_block_size: int = get_block_size(
            block_type="local_thresh", fov_dim=fov_dim, img_shape=img2mask_shape[0]
        )
        img2mask_thresh: np.ndarray = img2mask_blur > filters.threshold_local(
            img2mask_blur, block_size=local_thresh_block_size
        )

    # Remove small holes within the objects
    if hole_size is not None:
        img2mask_rm_holes: np.ndarray = morphology.remove_small_holes(
            img2mask_thresh, area_threshold=hole_size
        )
    else:
        small_holes_block_size: int = get_block_size(
            block_type="small_holes", fov_dim=fov_dim, img_shape=img2mask_shape[0]
        )
        img2mask_rm_holes: np.ndarray = morphology.remove_small_holes(
            img2mask_thresh, area_threshold=small_holes_block_size
        )

    # Filter projections
    if object_shape_type == "blob":
        img2mask_filtered: np.ndarray = img2mask_rm_holes
    else:
        img2mask_filtered: np.ndarray = filters.meijering(
            img2mask_rm_holes, sigmas=range(1, 5, 1), black_ridges=False
        )

    # Extract `label` and `area` from regionprops
    labeled_object_masks = measure.label(img2mask_filtered, connectivity=2)

    # Convert dictionary of region properties to DataFrame
    object_masks_df: Dict = pd.DataFrame(
        measure.regionprops_table(
            label_image=labeled_object_masks,
            cache=True,
            properties=[
                "label",
                "area",
            ],
        )
    )

    # Filter sizes: min_object_area < object_area < max_object_area
    filtered_objects = object_masks_df[
        object_masks_df["area"].between(
            left=min_object_area, right=max_object_area, inclusive="left"
        )
    ]

    # Map filtered objects to the object mask
    objects_mask_filtered_by_size = map_array(
        labeled_object_masks,
        object_masks_df["label"].to_numpy(),
        filtered_objects["label"].to_numpy(),
    )

    # TODO: update the integer conversion?
    return objects_mask_filtered_by_size.astype(int)


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
