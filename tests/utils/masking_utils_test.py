import os
import tempfile
import numpy as np
import pandas as pd

from alpineer.test_utils import create_paired_xarray_fovs, make_labels_xarray
from alpineer import image_utils
from ark.utils import masking_utils


def test_create_mask():
    test_img = np.array([[0, 2, 3, 0], [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # check for blur in masking
    mask = masking_utils.create_mask(test_img, intensity_thresh=1, sigma=1, min_mask_size=0,
                                     max_hole_size=0)
    assert np.all(
        np.equal(mask, np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))

    # check for removal of small objects
    processed_mask = masking_utils.create_mask(test_img, intensity_thresh=1, sigma=1,
                                               min_mask_size=5, max_hole_size=0)
    assert np.all(np.equal(processed_mask, np.zeros((4, 4))))


def test_create_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, arr = create_paired_xarray_fovs(
            temp_dir, fov_names=["fov1"], channel_names=["chan1", "chan2", "chan3"], fills=True)

        # check for single channel image
        single_channel_img = masking_utils.create_composite_image(temp_dir, "fov1", ["chan2"])
        assert np.all(single_channel_img == np.array(arr[0, :, :, 1]))

        # check for composite image
        composite_img = masking_utils.create_composite_image(
            temp_dir, "fov1",  ["chan2", "chan3"])
        assert np.all(composite_img == np.array(arr[0, :, :, 1])+np.array(arr[0, :, :, 2]))


def test_generate_img_masks():
    with tempfile.TemporaryDirectory() as temp_dir:
        img_dir = os.path.join(temp_dir, "image_data")
        mask_dir = os.path.join(temp_dir, "mask_dir")
        os.makedirs(img_dir)

        fovs = ["fov1", "fov2", "fov3"]
        _, _ = create_paired_xarray_fovs(
            img_dir, fov_names=fovs, channel_names=["chan1", "chan2"])

        masking_utils.generate_img_masks(
            img_dir, mask_dir, channels=["chan1", "chan2"], mask_name="composite_mask")

        for fov in fovs:
            assert os.path.exists(os.path.join(mask_dir, fov, "composite_mask.tiff"))


def test_create_cell_mask():
    segmentation_data = make_labels_xarray(
        None, fov_ids=["fov1"], compartment_names=['whole_cell'])
    seg_mask = np.array(segmentation_data[0, :, :, 0])

    cells = np.unique(seg_mask)
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1"] * len(cells),
            "label": cells,
            "cluster_name": [f"cluster_{cell}" for cell in cells]
        }
    )

    # single cell mask with no blurring
    exact_single_mask = masking_utils.create_cell_mask(
        seg_mask, cell_table, "fov1", cell_types=["cluster_1"], cluster_col="cluster_name",
        sigma=0, smooth_thresh=0.3)

    cluster_mask = seg_mask.copy()
    cluster_mask[cluster_mask > 1] = 0
    assert np.all(np.equal(cluster_mask, exact_single_mask))

    # multiple cell mask with no blurring
    exact_mask = masking_utils.create_cell_mask(
        seg_mask, cell_table, "fov1", cell_types=["cluster_1", "cluster_2"],
        cluster_col="cluster_name", sigma=0, smooth_thresh=0.3)

    cluster_mask = seg_mask.copy()
    cluster_mask[cluster_mask > 2] = 0
    cluster_mask[cluster_mask == 2] = 1
    assert np.all(np.equal(cluster_mask, exact_mask))


def test_generate_cell_masks():
    with tempfile.TemporaryDirectory() as temp_dir:
        seg_dir = os.path.join(temp_dir, "deepcell_output")
        mask_dir = os.path.join(temp_dir, "mask_dir")
        os.makedirs(seg_dir)

        fovs = ["fov1", "fov2", "fov3"]

        segmentation_data = make_labels_xarray(
            None, fov_ids=fovs, compartment_names=['whole_cell'])

        cell_table = []
        for i, fov in enumerate(fovs):
            cells = np.unique(segmentation_data[i, :, :, 0])
            fov_table = pd.DataFrame(
                {
                    "fov": ["fov1"] * len(cells),
                    "label": cells,
                    "cluster_name": [f"cluster_{cell}" for cell in cells]
                }
            )
            cell_table.append(fov_table)

            img_path = os.path.join(seg_dir, fov)
            image_utils.save_image(img_path + "_whole_cell.tiff", segmentation_data[i, :, :, 0])

        cell_table = pd.concat(cell_table)

        masking_utils.generate_cell_masks(
            seg_dir, mask_dir, cell_table, ["cluster_1"], "cluster_name", mask_name="cell_mask")

        for fov in fovs:
            assert os.path.exists(os.path.join(mask_dir, fov, "cell_mask.tiff"))
