import numpy as np
import os
import pathlib
import pytest
import skimage.io as io
import xarray as xr

from alpineer.load_utils import load_imgs_from_tree
from ark.segmentation.ez_seg import composites


@pytest.fixture(scope="session")
def image_dir(tmpdir_factory: pytest.TempPathFactory) -> pathlib.Path:
    image_dir_name: pathlib.Path = tmpdir_factory.mktemp("image_dir")
    fovs: List[str] = [f"fov{i}" for i in np.arange(3)]
    chans: List[str] = [f"chan{i}" for i in np.arange(2)]

    example_img_0 = np.array(
        [[0] * 4,
         [1] * 4,
         [2] * 4,
         [3] * 4]
    )
    example_img_1 = np.array(
        [[0, 0, 1, 1],
         [1, 1, 2, 2],
         [2, 2, 3, 3],
         [3, 3, 4, 4]]
    )
    example_imgs = [example_img_0, example_img_1]

    for fov in fovs:
        fov_dir: pathlib.Path = image_dir_name / fov
        os.mkdir(fov_dir)
        for i, chan in enumerate(chans):
            io.imsave(str(fov_dir / chan + ".tiff"), example_imgs[i])

    yield image_dir_name


@pytest.fixture(scope="session")
def image_data(image_dir: pathlib.Path) -> xr.DataArray:
    yield load_imgs_from_tree(
        data_dir=image_dir, img_sub_folder=None, fovs=["fov0"]
    )


@pytest.fixture(scope="session")
def composite_array_add() -> np.ndarray:
    yield np.array(
        [[0] * 4,
         [1] * 4,
         [2] * 4,
         [3] * 4]
    )


@pytest.fixture(scope="session")
def composite_array_subtract() -> np.ndarray:
    yield np.array(
        [[3] * 4,
         [2] * 4,
         [1] * 4,
         [0] * 4]
    )


def test_add_to_composite_signal(image_data: xr.DataArray, composite_array_add: np.ndarray):
    composite_array_added: np.ndarray = composites.add_to_composite(
        data=image_data,
        composite_array=composite_array_add,
        images_to_add=["chan0", "chan1"],
        image_type="signal",
        composite_method="total"
    )

    result: np.ndarray = np.array(
        [[0, 0, 1, 1],
         [2, 2, 3, 3],
         [4, 4, 5, 5],
         [6, 6, 7, 7]]
    )
    assert np.all(composite_array_added == result)


def test_add_to_composite_signal_binary(image_data: xr.DataArray, composite_array_add: np.ndarray):
    composite_array_added: np.ndarray = composites.add_to_composite(
        data=image_data,
        composite_array=composite_array_add,
        images_to_add=["chan0", "chan1"],
        image_type="signal",
        composite_method="binary"
    )

    result: np.ndarray = np.array(
        [[0, 0, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    )
    assert np.all(composite_array_added == result)


def test_add_to_composite_pixel_cluster(image_data: xr.DataArray, composite_array_add: np.ndarray):
    composite_array_added: np.ndarray = composites.add_to_composite(
        data=image_data,
        composite_array=composite_array_add,
        images_to_add=["chan0"],
        image_type="pixel_cluster",
        composite_method="binary"
    )

    result: np.ndarray = np.array(
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    )
    assert np.all(composite_array_added == result)


def test_subtract_from_composite_signal_binary(
    image_data: xr.DataArray, composite_array_subtract: np.ndarray
):
    composite_array_subtracted: np.ndarray = composites.subtract_from_composite(
        data=image_data,
        composite_array=composite_array_subtract.copy(),
        images_to_subtract=["chan0", "chan1"],
        image_type="signal",
        composite_method="binary"
    )

    result: np.ndarray = np.array(
        [[1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    )
    assert np.all(composite_array_subtracted == result)


def test_subtract_from_composite_general(
    image_data: xr.DataArray, composite_array_subtract: np.ndarray
):
    # also handles other casees that aren't image_type="signal" + composite_method="binary"
    composite_array_subtracted: np.ndarray = composites.subtract_from_composite(
        data=image_data,
        composite_array=composite_array_subtract.copy(),
        images_to_subtract=["chan1"],
        image_type="pixel_cluster",
        composite_method="total"
    )

    result: np.ndarray = np.array(
        [[3, 3, 2, 2],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    )
    assert np.all(composite_array_subtracted == result)
