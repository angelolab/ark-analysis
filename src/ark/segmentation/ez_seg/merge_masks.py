import pathlib
from typing import List, Union
import xarray as xr
import numpy as np
import os
from skimage.io import imread
from skimage.morphology import label
from skimage.measure import regionprops
from alpineer import load_utils, image_utils
from ark.segmentation.ez_seg.ez_seg_utils import log_creator


def merge_masks_seq(
    fov_list: List[str],
    object_list: List[str],
    object_mask_dir: Union[pathlib.Path, str],
    cell_mask_dir: Union[pathlib.Path, str],
    cell_mask_suffix: str,
    overlap_percent_threshold: int,
    first_merge: bool,
    operation_type: str,
    save_path_merge: Union[pathlib.Path, str],
    save_path_remain: Union[pathlib.Path, str],
    log_dir: Union[pathlib.Path, str]
) -> None:
    """
    Sequentially merge object masks with cell masks. Object list is ordered enforced, e.g. object_list[i] will merge
    overlapping object masks with cell masks from the initial cell segmentation. Remaining, un-merged cell masks will
    then be used to merge with object_list[i+1], etc.

    Args:
        fov_list (List[str]): A list of fov names to merge masks over.
        object_list (List[str]): A name representing previously generated object masks. Note, order matters.
        object_mask_dir (Union[pathlib.Path, str]): Directory where object (ez) segmented masks are located
        cell_mask_dir (Union[str, pathlib.Path]): Path to where the original cell masks are located.
        cell_mask_suffix (str): Name of the cell type you are merging. Usually "whole_cell".
        overlap_percent_threshold (int): Percent overlap of cell's total pixel area needed for object merge.
        first_merge (bool): Whether or not this is the first merge performed in a dataset.
        operation_type (str): Action performed - either combine objects with cells or remove overlapping cells.
        save_path_merge (Union[str, pathlib.Path]): Directory where combined masks or original objects will be saved.
        save_path_remain: (Union[pathlib.Path, str]): Directory where remaining cell / object mask will be saved.
        log_dir (Union[str, pathlib.Path]): The directory to save log information to.
    """
    # validate paths
    if isinstance(object_mask_dir, str):
        object_mask_dir = pathlib.Path(object_mask_dir)

    # if first_merge is True, cell mask path is the original cell dir,
    # Otherwise the cell masks to be used will come from the remain_cell_dir.
    if first_merge is True:
        if isinstance(cell_mask_dir, str):
            cell_mask_dir = pathlib.Path(cell_mask_dir)
    elif first_merge is False:
        if isinstance(save_path_remain, str):
            cell_mask_dir = pathlib.Path(save_path_remain)

    if isinstance(save_path_merge, str):
        save_path_merge = pathlib.Path(save_path_merge)
    if isinstance(save_path_remain, str):
        save_path_remain = pathlib.Path(save_path_remain)

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
            remaining_cells = merge_masks_single(
                object_mask=curr_object_mask,
                cell_mask=curr_cell_mask,
                overlap_thresh=overlap_percent_threshold,
                operation=operation_type,
                object_name=obj,
                mask_save_path=save_path_merge,
            )
            curr_cell_mask = remaining_cells

        # save the unmerged cells as a tiff.
        image_utils.save_image(fname=save_path_remain / (fov + f"_{cell_mask_suffix}.tiff"), data=curr_cell_mask.astype(np.int32))

    # Write a log saving mask merging info
    variables_to_log = {
        "fov_list": fov_list,
        "object_list": object_list,
        "object_mask_dir": object_mask_dir,
        "cell_mask_dir": cell_mask_dir,
        "cell_mask_suffix": cell_mask_suffix,
        "overlap_percent_threshold": overlap_percent_threshold,
        "operation_type": operation_type,
        "save_path_merge": save_path_merge,
        "save_path_remain": save_path_remain
    }
    log_creator(variables_to_log, log_dir, "mask_merge_log.txt")
    if operation_type == "combine":
        print(f"Combined object and cell masks built and saved, Remaining cell masks saved.")
    elif operation_type == "remove_duplicates":
        print("Duplicate masks removed. Original objects and remaining cells saved.")


def merge_masks_single(
    object_mask: np.ndarray,
    cell_mask: np.ndarray,
    overlap_thresh: int,
    operation: str,
    object_name: str,
    mask_save_path: str
) -> np.ndarray:
    """
    Combines overlapping object and cell masks. For any combination which represents has at least `overlap` percentage
    of overlap, the combined mask is kept and incorporated into the original object masks to generate a new set of masks.

    Args:
        object_mask (np.ndarray): The object mask numpy array.
        cell_mask (np.ndarray): The cell mask numpy array.
        overlap_thresh (int): The percentage overlap required for a cell to be merged.
        operation (str): Action performed by function - either combine objects with cells or remove overlapping cells.
        object_name (str): The name of the object.
        mask_save_path (str): The path to save the mask.

    Returns:
        np.ndarray: The cells remaining mask, which will be used for the next cycle in merging while there are objects.
        When no more cells and objects are left to merge, the final, non-merged cells are returned.
    """

    if cell_mask.shape != object_mask.shape:
        raise ValueError("Both masks must have the same shape")

    # Relabel cell, object masks
    cell_labels, num_cell_labels = label(cell_mask, return_num=True)
    object_labels, num_object_labels = label(object_mask, return_num=True)

    # Instantiate new array for merging (or keeping original object masks if removing duplicates).
    merged_mask = object_labels.copy()

    # Set up list to store merged cell labels
    remove_cells_list = [0]

    # Create a dictionary of the bounding boxes for all object labels
    object_labels_bounding_boxes = get_bounding_boxes(object_labels)

    # Find connected components in object and cell masks. Merge only those with highest overlap that meets threshold.
    for obj_label in range(1, num_object_labels + 1):
        # Extract a connected component from object_mask
        object_mask_component = object_labels == obj_label

        best_overlap = 0
        best_cell_mask_component = None
        cell_to_merge_label = None

        # Filter for cell_labels that fall within the expanded bounding box of the obj_label
        cell_labels_in_range = filter_labels_in_bbox(object_labels_bounding_boxes[obj_label], cell_labels)

        for cell_label in cell_labels_in_range:
            # Extract a connected component from cell_mask
            cell_mask_component = cell_labels == int(cell_label)

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

        # If best merge has been found, assign the combined cell+object into the new mask and record the cell label.
        if operation == "combine" and best_cell_mask_component is not None:
            merged_mask[best_cell_mask_component == True] = obj_label
            remove_cells_list.append(cell_to_merge_label)

        # If best merge has been found, assign the object into the new mask and record the cell label for later removal.
        elif operation == "remove_duplicates" and best_cell_mask_component is not None:
            remove_cells_list.append(cell_to_merge_label)

    # Assign any cells not combined or removed into a remaining cell mask array.
    non_merged_cell_mask = np.isin(cell_labels, remove_cells_list, invert=True)
    cell_labels[non_merged_cell_mask == False] = 0

    # Save the merged mask tiff.
    if operation == "combine":
        filename = os.path.join(mask_save_path, object_name.removesuffix(".tiff") + "_combined.tiff")
    elif operation == "remove_duplicates":
        filename = os.path.join(mask_save_path, object_name.removesuffix(".tiff") + "_duplicates_removed.tiff")
    image_utils.save_image(
        fname=filename,
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

    props = regionprops(object_labels)

    for prop in props:
        # Get major axis length
        major_axis_length = prop.major_axis_length

        # Define bounding box based on major axis length
        centroid = prop.centroid
        radius = int(major_axis_length / 2)
        min_row = max(0, int(centroid[0]) - radius)
        max_row = min(object_labels.shape[0] - 1, int(centroid[0]) + radius)
        min_col = max(0, int(centroid[1]) - radius)
        max_col = min(object_labels.shape[1] - 1, int(centroid[1]) + radius)

        bounding_boxes[prop.label] = ((min_row, min_col), (max_row, max_col))

    return bounding_boxes


def filter_labels_in_bbox(bounding_box: List, cell_labels: np.ndarray):
    """
    Gets the cell labels that fall within the expanded bounding box of a given object.

    Args:
        bounding_box (List): The bounding box values for the input obj_label
        cell_labels (np.ndarray): The cell label array.

    Returns:
        List: The cell labels that fall within the expanded bounding box.

    """
    min_row, min_col = bounding_box[0]
    max_row, max_col = bounding_box[1]

    filtered_labels = []

    props = regionprops(cell_labels)

    for prop in props:
        centroid = prop.centroid
        if min_row <= centroid[0] <= max_row and min_col <= centroid[1] <= max_col:
            filtered_labels.append(prop.label)

    return filtered_labels
