import pathlib
from typing import List, Union
import xarray as xr
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops_table, label
from alpineer import load_utils, image_utils


def merge_masks(
    object_mask: np.ndarray,
    cell_mask: np.ndarray,
    overlap: int,
    object_name: str,
    mask_save_path: str,
) -> np.ndarray:
    """
    Combines overlapping object and cell masks. For any combination which represents has atleast `overlap` percentage
    of overlap, the combined mask is kept and incorporated into the original object masks to generate a new set of masks.

    Args:
        object_mask (np.ndarray): The object mask numpy array.
        cell_mask (np.ndarray): The cell mask numpy array.
        overlap (int): The amount of overlap required for a cell to be merged.
        object_name (str): The name of the object.
        mask_save_path (str): The path to save the mask.

    Returns:
        np.ndarray: The cells remaining mask, which will be used for the next cycle in merging while there are objects.
        When no more cells and objects are left to merge, the final, non-merged cells are returned.
    """
    # combine object and cell images using 'and' producing a binary overlap image
    binary_overlap_image = np.bitwise_and(object_mask.astype(np.uint8), cell_mask)
    binary_overlap_image = binary_overlap_image > 0
    # label the cell mask image
    cell_mask_labels = label(cell_mask)
    # using regionprops_table, input binary overlap image as intensity data and cell label image as label image
    find_cells_table = regionprops_table(
        label_image=cell_mask_labels,
        intensity_image=binary_overlap_image,
        properties=["label", "intensity_mean"],
    )
    # convert table into pandas data frame
    find_cells_df = pd.DataFrame(find_cells_table)
    # any cell properties with mean intensity over the threshold of positivity (per_overlap), keep. Discard the rest.
    cells_to_merge = find_cells_df[find_cells_df.intensity_mean > overlap / 100]
    # Create image with only the filtered overlapping cells
    cells_to_merge_mask = np.isin(cell_mask_labels, cells_to_merge["label"])
    # combine images into one and relabel. Save image with new labels as named object + cell mask merges.
    final_overlap_image = np.bitwise_and(object_mask, cells_to_merge_mask)
    image_utils.save_image(
        fname=mask_save_path / (object_name + "_merged.tiff"),
        data=final_overlap_image)

    # save cell masks without the masks that have been incorporated into the merged selections
    cells_to_keep_mask = np.isin(cell_mask_labels, cells_to_merge["label"], invert=True)

    return cells_to_keep_mask


def merge_masks_seq(
    object_list: List[str],
    object_mask_dir: Union[pathlib.Path, str],
    cell_mask_path: Union[pathlib.Path, str],
    overlap: int,
    save_path: Union[pathlib.Path, str],
) -> None:
    """
    Sequentially merge object masks with cell masks. Object list is ordered envorced, e.g. object_list[i] will merge overlapping
    object masks with cell masks from the initial cell segmentation. Remaining, un-merged cell masks will then be used to merge with
    object_list[i+1], etc.

    Args:
        object_list (List[str]): A list of names representing previously generated object masks. Note, order matters.
        cell_mask_path (Union[str, pathlib.Path]): Path to where the original cell mask is located.
        overlap (int): Percent overlap of total pixel area needed for an object to be merged to a cell.
        save_path (Union[str, pathlib.Path]): The directory where the merged masks and remaining cell mask will be saved.
    """

    if isinstance(object_mask_dir, str):
        object_mask_dir = pathlib.Path(object_mask_dir)
    if isinstance(cell_mask_path, str):
        cell_mask_path = pathlib.Path(cell_mask_path)
    if isinstance(save_path, str):
        save_path = pathlib.Path(save_path)

    whole_cell_mask = imread(fname=cell_mask_path)

    objects: xr.DataArray = load_utils.load_imgs_from_dir(
        object_mask_dir, files=[o + ".tiff" for o in object_list]).drop_vars("compartments").squeeze()

    # sort the imported objects w.r.t the object_list
    objects.reindex(indexers={
        "fovs": object_list
    })

    for obj in object_list:
        curr_object_mask = imread(fname=(object_mask_dir / obj).with_suffix(".tiff"))
        remaining_cells = merge_masks(
            object_mask=curr_object_mask,
            cell_mask=whole_cell_mask,
            overlap=overlap,
            object_name=obj,
            mask_save_path=save_path,
        )
        curr_cell_mask = remaining_cells

    image_utils.save_image(fname=save_path / "final_cells_remaining.tiff", data=curr_cell_mask.astype(np.uint8))
