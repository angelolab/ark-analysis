import numpy as np
import os
import pathlib
import skimage.io as io
import tempfile
import xarray as xr

from alpineer import io_utils
from ark.segmentation.ez_seg import merge_masks
from scipy.ndimage import label
from skimage.draw import disk
from typing import List, Union


def test_merge_masks_seq():
    fov_list: List[str] = [f"fov{i}" for i in range(3)]
    object_list: List[str] = [f"mask{i}" for i in range(2)]

    with tempfile.TemporaryDirectory() as td:
        object_mask_dir: Union[str, pathlib.Path] = os.path.join(td, "ez_seg_dir")
        cell_mask_dir: Union[str, pathlib.Path] = os.path.join(td, "deepcell_output")
        merged_mask_dir: Union[str, pathlib.Path] = os.path.join(td, "merged_masks_dir")
        log_dir: Union[str, pathlib.Path] = os.path.join(td, "log_dir")
        for directory in [object_mask_dir, cell_mask_dir, merged_mask_dir, log_dir]:
            os.mkdir(directory)

        overlap_thresh: int = 10

        for fov in fov_list:
            cell_mask_data: np.ndarray = np.random.randint(0, 16, (32, 32))
            cell_mask_fov_file: Union[str, pathlib.Path] = os.path.join(
                cell_mask_dir, f"{fov}_whole_cell.tiff"
            )
            io.imsave(cell_mask_fov_file, cell_mask_data)

            for obj in object_list:
                object_mask_data: np.ndarray = np.random.randint(0, 8, (32, 32))
                object_mask_fov_file: Union[str, pathlib.Path] = os.path.join(
                    object_mask_dir, f"{fov}_{obj}.tiff"
                )
                io.imsave(object_mask_fov_file, cell_mask_data)

        # we're only testing functionality, for in-depth merge testing see test_merge_masks_single
        merge_masks.merge_masks_seq(
            fov_list, object_list, object_mask_dir, cell_mask_dir, overlap_thresh,
            merged_mask_dir, log_dir
        )

        for fov in fov_list:
            merged_mask_fov_file: Union[str, pathlib.Path] = os.path.join(
                merged_mask_dir, f"{fov}_final_cells_remaining.tiff"
            )
            assert os.path.exists(merged_mask_fov_file)

        log_file: Union[str, pathlib.Path] = os.path.join(log_dir, "mask_merge_log.txt")
        assert os.path.exists(log_file)

        with open(log_file) as infile:
            log_data: List[str] = infile.readlines()

        assert log_data[0] == f"fov_list: {str(fov_list)}\n"
        assert log_data[1] == f"object_list: {str(object_list)}\n"
        assert log_data[2] == f"object_mask_dir: {str(object_mask_dir)}\n"
        assert log_data[3] == f"cell_mask_path: {str(cell_mask_dir)}\n"
        assert log_data[4] == f"overlap_percent_threshold: {str(overlap_thresh)}\n"
        assert log_data[5] == f"save_path: {str(merged_mask_dir)}\n"


def test_merge_masks_single():
    object_mask: np.ndarray = np.zeros((32, 32))
    cell_mask: np.ndarray = np.zeros((32, 32))
    expected_merged_mask: np.ndarray = np.zeros((32, 32))
    expected_cell_mask: np.ndarray = np.zeros((32, 32))

    overlap_thresh: int = 10
    merged_mask_name: str = "merged_mask"

    # case 1: overlap below threshold, don't merge
    obj1_rows, obj1_cols = disk((7, 7), radius=5, shape=object_mask.shape)
    cell1_rows, cell1_cols = disk((1, 1), radius=5, shape=cell_mask.shape)
    cell2_rows, cell2_cols = disk((13, 13), radius=5, shape=cell_mask.shape)
    object_mask[obj1_rows, obj1_cols] = 1
    cell_mask[cell1_rows, cell1_cols] = 1
    cell_mask[cell2_rows, cell2_cols] = 2

    # case 2: multiple cells within threshold, only merge best one
    obj2_rows, obj2_cols = disk((25, 25), radius=5, shape=object_mask.shape)
    cell3_rows, cell3_cols = disk((20, 20), radius=5, shape=cell_mask.shape)
    cell4_rows, cell4_cols = disk((27, 27), radius=5, shape=cell_mask.shape)
    object_mask[obj2_rows, obj2_cols] = 2
    cell_mask[cell3_rows, cell3_cols] = 3
    cell_mask[cell4_rows, cell4_cols] = 4

    expected_merged_mask[obj1_rows, obj1_cols] = 1
    expected_merged_mask[obj2_rows, obj2_cols] = 2
    expected_merged_mask[cell4_rows, cell4_cols] = 2

    expected_cell_mask[cell1_rows, cell1_cols] = 1
    expected_cell_mask[cell2_rows, cell2_cols] = 2
    expected_cell_mask[cell3_rows, cell3_cols] = 3

    with tempfile.TemporaryDirectory() as td:
        mask_save_dir: Union[str, pathlib.Path] = os.path.join(td, "mask_save_dir")
        os.mkdir(mask_save_dir)

        created_cell_mask: np.ndarray = merge_masks.merge_masks_single(
            object_mask, cell_mask, overlap_thresh, merged_mask_name, mask_save_dir
        )

        created_merged_mask: np.ndarray = io.imread(
            os.path.join(mask_save_dir, merged_mask_name + "_merged.tiff")
        )

        assert np.all(created_merged_mask == expected_merged_mask)
        assert np.all(created_cell_mask == expected_cell_mask)
