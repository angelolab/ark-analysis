import pathlib
from dataclasses import dataclass

import numpy as np
import pytest
from alpineer import image_utils
from skimage.io import imread

from ark.segmentation.ez_seg import ez_seg_display


@dataclass
class MaskDataPaths:
    image_data_path: pathlib.Path
    mask_path: pathlib.Path
    fov0_dir: pathlib.Path
    object_mask_dir: pathlib.Path
    cell_mask_dir: pathlib.Path
    merged_mask_dir: pathlib.Path


@pytest.fixture(scope="module")
def mask_data(
    tmp_path_factory: pytest.TempPathFactory, rng: np.random.Generator
) -> tuple[pathlib.Path, pathlib.Path]:
    img_data_path = tmp_path_factory.mktemp("image_data")
    mask_path = tmp_path_factory.mktemp("mask_data")

    fov0 = "fov_0"
    mask0 = "mask_0"

    # Create directories
    fov0_dir = img_data_path / fov0
    object_mask_dir = mask_path / "object_mask_dir"
    cell_mask_dir = mask_path / "cell_mask_dir"
    merged_mask_dir = mask_path / "merged_mask_dir"

    for p in [fov0_dir, object_mask_dir, cell_mask_dir, merged_mask_dir]:
        p.mkdir(parents=True, exist_ok=True)

    fov0_chan0_img = fov0_dir / "chan_0.tiff"
    object_mask_img = object_mask_dir / f"{fov0}_{mask0}.tiff"
    cell_mask_img = cell_mask_dir / f"{fov0}_whole_cell.tiff"
    merged_mask_img = merged_mask_dir / f"{fov0}_{mask0}_merged.tiff"

    image_utils.save_image(fname=fov0_chan0_img, data=rng.random(size=(1024, 1024)))
    image_utils.save_image(fname=object_mask_img, data=rng.random(size=(1024, 1024)))
    image_utils.save_image(fname=cell_mask_img, data=rng.random(size=(1024, 1024)))
    image_utils.save_image(fname=merged_mask_img, data=rng.random(size=(1024, 1024)))

    yield MaskDataPaths(
        image_data_path=img_data_path,
        mask_path=mask_path,
        fov0_dir=fov0_dir,
        object_mask_dir=object_mask_dir,
        cell_mask_dir=cell_mask_dir,
        merged_mask_dir=merged_mask_dir,
    )


def test_display_channel_image(mask_data: MaskDataPaths):
    ez_seg_display.display_channel_image(
        base_image_path=mask_data.image_data_path,
        sub_folder_name=None,
        test_fov_name="fov_0",
        channel_name="chan_0",
    )

    with pytest.raises(FileNotFoundError):
        ez_seg_display.display_channel_image(
            base_image_path=mask_data.image_data_path,
            sub_folder_name=None,
            test_fov_name="fov_0",
            channel_name="bad_chan_name",
        )


def test_overlay_mask_outlines(mask_data: MaskDataPaths):
    ez_seg_display.overlay_mask_outlines(
        fov="fov_0",
        channel="chan_0",
        image_dir=mask_data.image_data_path,
        sub_folder_name=None,
        mask_name="mask_0",
        mask_dir=mask_data.object_mask_dir,
    )

    with pytest.raises(FileNotFoundError):
        ez_seg_display.overlay_mask_outlines(
            fov="fov_0",
            channel="chan_0",
            image_dir=mask_data.image_data_path,
            sub_folder_name=None,
            mask_name="bad_mask_name",
            mask_dir=mask_data.object_mask_dir,
        )

    with pytest.raises(FileNotFoundError):
        ez_seg_display.overlay_mask_outlines(
            fov="fov_0",
            channel="bad_chan_name",
            image_dir=mask_data.image_data_path,
            sub_folder_name=None,
            mask_name="mask_0",
            mask_dir=mask_data.object_mask_dir,
        )


def test_multiple_mask_display(mask_data: MaskDataPaths):
    fov0 = "fov_0"
    mask0 = "mask_0"

    ez_seg_display.multiple_mask_display(
        fov=fov0,
        mask_name=mask0,
        object_mask_dir=mask_data.object_mask_dir,
        cell_mask_dir=mask_data.cell_mask_dir,
        merged_mask_dir=mask_data.merged_mask_dir,
    )

    with pytest.raises(FileNotFoundError):
        ez_seg_display.multiple_mask_display(
            fov=fov0,
            mask_name="bad_mask_name",
            object_mask_dir=mask_data.object_mask_dir,
            cell_mask_dir=mask_data.cell_mask_dir,
            merged_mask_dir=mask_data.merged_mask_dir,
        )


def test_create_overlap_and_merge_visual(mask_data: MaskDataPaths):
    overlap_visual: np.ndarray = ez_seg_display.create_overlap_and_merge_visual(
        fov="fov_0",
        mask_name="mask_0",
        object_mask_dir=mask_data.object_mask_dir,
        cell_mask_dir=mask_data.cell_mask_dir,
        merged_mask_dir=mask_data.merged_mask_dir,
    )

    assert (
        overlap_visual.shape[:2]
        == imread(mask_data.image_data_path / "fov_0" / "chan_0.tiff").shape
    )
    assert overlap_visual.shape[-1] == 3  # rgb channels
    assert overlap_visual.dtype == np.uint8
