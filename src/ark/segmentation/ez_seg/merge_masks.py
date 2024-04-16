import pathlib
from typing import List, Union
import xarray as xr
import numpy as np
import os
from skimage.io import imread
from skimage.morphology import label
from skimage.measure import regionprops_table
import pandas as pd
from alpineer import load_utils, image_utils
from ark.segmentation.ez_seg.ez_seg_utils import log_creator


def merge_masks_seq(
    fov_list: List[str],
    object_list: List[str],
    object_mask_dir: Union[pathlib.Path, str],
    cell_mask_dir: Union[pathlib.Path, str],
    cell_mask_suffix: str,
    overlap_percent_threshold: int,
    expansion_factor: int,
    save_path: Union[pathlib.Path, str],
    log_dir: Union[pathlib.Path, str]
) -> None:
    """
    Sequentially merge object masks with cell masks. Object list is ordered enforced, e.g. object_list[i] will merge
    overlapping object masks with cell masks from the initial cell segmentation. Remaining, un-merged cell masks will
    then be used to merge with object_list[i+1], etc.

    Args:
        fov_list (List[str]): A list of fov names to merge masks over.
        object_list (List[str]): A list of names representing previously generated object masks. Note, order matters.
        object_mask_dir (Union[pathlib.Path, str]): Directory where object (ez) segmented masks are located
        cell_mask_dir (Union[str, pathlib.Path]): Path to where the original cell masks are located.
        cell_mask_suffix (str): Name of the cell type you are merging. Usually "whole_cell".
        overlap_percent_threshold (int): Percent overlap of total pixel area needed fo object to be merged to a cell.
        expansion_factor (int): How many pixels out from an objects bbox a cell should be looked for.
        save_path (Union[str, pathlib.Path]): The directory where merged masks and remaining cell mask will be saved.
        log_dir (Union[str, pathlib.Path]): The directory to save log information to.
    """
    # validate paths
    if isinstance(object_mask_dir, str):
        object_mask_dir = pathlib.Path(object_mask_dir)
    if isinstance(cell_mask_dir, str):
        cell_mask_dir = pathlib.Path(cell_mask_dir)
    if isinstance(save_path, str):
        save_path = pathlib.Path(save_path)

    # for each fov, import cell and object masks (multiple mask types into single xr.DataArray)
    for fov in fov_list:
        curr_cell_mask = imread(fname=os.path.join(
            cell_mask_dir, '_'.join([f'{fov}', f'{cell_mask_suffix}.tiff']))
        )

        fov_object_names = [f'{fov}_' + obj + '.tiff' for obj in object_list]

        objects: xr.DataArray = load_utils.load_imgs_from_dir(
            object_mask_dir, files=fov_object_names).drop_vars("compartments").squeeze()

        # sort the imported objects w.r.t the object_list
        objects.reindex(indexers={
            "fovs": fov_object_names
        })

        # for each object type in the fov, merge with cell masks
        for obj in fov_object_names:
            curr_object_mask = imread(fname=(object_mask_dir / obj))
            remaining_cells = merge_masks_single(object_mask=curr_object_mask, cell_mask=curr_cell_mask,
                                                 overlap_thresh=overlap_percent_threshold, object_name=obj,
                                                 mask_save_path=save_path, expansion_factor=expansion_factor)
            curr_cell_mask = remaining_cells

        # save the unmerged cells as a tiff.
        image_utils.save_image(fname=save_path / (fov + f"_final_{cell_mask_suffix}_remaining.tiff"), data=curr_cell_mask.astype(np.int32))

    # Write a log saving mask merging info
    variables_to_log = {
        "fov_list": fov_list,
        "object_list": object_list,
        "object_mask_dir": object_mask_dir,
        "cell_mask_dir": cell_mask_dir,
        "cell_mask_suffix": cell_mask_suffix,
        "overlap_percent_threshold": overlap_percent_threshold,
        "save_path": save_path
    }
    log_creator(variables_to_log, log_dir, "mask_merge_log.txt")
    print("Merged masks built and saved")


def merge_masks_single(
    object_mask: np.ndarray,
    cell_mask: np.ndarray,
    overlap_thresh: int,
    object_name: str,
    mask_save_path: str,
    expansion_factor: int
) -> np.ndarray:
    """
    Combines overlapping object and cell masks. For any combination which represents has at least `overlap` percentage
    of overlap, the combined mask is kept and incorporated into the original object masks to generate a new set of masks.

    Args:
        object_mask (np.ndarray): The object mask numpy array.
        cell_mask (np.ndarray): The cell mask numpy array.
        overlap_thresh (int): The percentage overlap required for a cell to be merged.
        object_name (str): The name of the object.
        mask_save_path (str): The path to save the mask.
        expansion_factor (int): How many pixels out from an objects bbox a cell should be looked for.

    Returns:
        np.ndarray: The cells remaining mask, which will be used for the next cycle in merging while there are objects.
        When no more cells and objects are left to merge, the final, non-merged cells are returned.
    """

    if cell_mask.shape != object_mask.shape:
        raise ValueError("Both masks must have the same shape")

    # Relabel cell, object masks
    cell_labels, num_cell_labels = label(cell_mask, return_num=True)
    object_labels, num_object_labels = label(object_mask, return_num=True)

    # Instantiate new array for merging
    merged_mask = object_labels.copy()

    # Set up list to store merged cell labels
    remove_cells_list = [0]

    # Create a dictionary of the bounding boxes for all object labels
    object_labels_bounding_boxes = get_bounding_boxes(object_labels)

    # Calculate all cell regionprops for filtering, convert to DataFrame
    cell_props = pd.DataFrame(regionprops_table(cell_labels, properties=('label', 'centroid')))

    # Find connected components in object and cell masks. Merge only those with highest overlap that meets threshold.
    for obj_label in range(1, num_object_labels + 1):
        # Extract a connected component from object_mask
        object_mask_component = object_labels == obj_label

        best_overlap = 0
        best_cell_mask_component = None
        cell_to_merge_label = None

        # Filter for cell_labels that fall within the expanded bounding box of the obj_label
        cell_labels_in_range = filter_labels_in_bbox(
            object_labels_bounding_boxes[obj_label], cell_props, expansion_factor)

        for cell_label in cell_labels_in_range:
            # Extract a connected component from cell_mask
            cell_mask_component = cell_labels == cell_label

            # Calculate the overlap between cell_mask_component and object_mask_component
            intersection = np.logical_and(cell_mask_component, object_mask_component)
            overlap = intersection.sum()

            # Calculate cell-object overlap percent threshold
            meets_overlap_thresh = overlap / cell_mask_component.sum() > overlap_thresh / 100

            # Ensure cell overlap meets percent threshold and has the highest relative cell-object overlap
            if overlap > best_overlap and meets_overlap_thresh:
                best_overlap = overlap
                best_cell_mask_component = cell_mask_component
                cell_to_merge_label = cell_label

        # If best merge has been found, assign the merged cell+object into the new mask and record the cell label
        if best_cell_mask_component is not None:
            merged_mask[best_cell_mask_component == True] = obj_label
            remove_cells_list.append(cell_to_merge_label)

    # Assign any unmerged cells into a remaining cell mask array.
    non_merged_cell_mask = np.isin(cell_labels, remove_cells_list, invert=True)
    cell_labels[non_merged_cell_mask == False] = 0

    # Save the merged mask tiff.
    image_utils.save_image(
        fname=os.path.join(mask_save_path, object_name.removesuffix(".tiff") + "_merged.tiff"),
        data=merged_mask)

    # Return unmerged cells
    return cell_labels


def get_bounding_boxes(object_labels: np.ndarray):
    """
    Gets the bounding boxes of labeled images based on object major axis length.

    Args:
        object_labels (np.ndarray): label array
    Returns:
        dict: Dictionary containing labels as keys and bounding box as values
    """
    bounding_boxes = {}

    # Get region properties as a DataFrame
    props_df = pd.DataFrame(regionprops_table(object_labels, properties=('label', 'bbox')))

    # Return closed interval bounding box
    # label_id, min_row, min_col, max_row, max_col used to define bbox
    props_df.apply(lambda row: bounding_boxes.update(
        {row['label']: ((row['bbox-0'], row['bbox-1']), (row['bbox-2'] - 1, row['bbox-3'] - 1))}), axis=1)

    return bounding_boxes


def filter_labels_in_bbox(bounding_box: List, cell_props: pd.DataFrame, expansion_factor: int):
    """
    Gets the cell labels that fall within the expanded bounding box of a given object.

    Args:
        bounding_box (List): The bounding box values for the input obj_label
        cell_props (pd.DataFrame): The cell label regionprops DataFrame.
        expansion_factor: how many pixels from the bounding box you want to expand the search for compatible cells.

    Returns:
        List: The cell labels that fall within the expanded bounding box.

    """
    min_row, min_col = bounding_box[0]
    max_row, max_col = bounding_box[1]

    # Filter labels based on bounding box
    filtered_labels = cell_props[(cell_props['centroid-0'] >= min_row-expansion_factor) &
                                 (cell_props['centroid-0'] <= max_row+expansion_factor) &
                                 (cell_props['centroid-1'] >= min_col-expansion_factor) &
                                 (cell_props['centroid-1'] <= max_col+expansion_factor)]['label'].tolist()

    return filtered_labels

