import numpy as np
import os
import pathlib
import pytest
import skimage.io as io
import tempfile
import xarray as xr

from alpineer import image_utils, io_utils
from ark.segmentation.ez_seg import ez_seg_utils
from pytest_cases import param_fixture
from scipy import ndimage
from skimage import draw
from typing import List, Tuple, Union


# @pytest.fixture(scope="module")
# def ez_fov(
#     tmpdir_factory: pytest.TempPathFactory, rng: np.random.Generator
# ) -> pathlib.Path:
#     """
#     Creates an DataArray with a 1024 by 1024 image with random noise, uneven illumination
#     and spots.

#     Yields:
#         pathlib.Path: The path to the FOV.
#     """
#     channel_count: int = 3
#     image_size: int = 1024
#     spot_count: int = 60
#     spot_radius: int = 40
#     cloud_noise_size: int = 4

#     image: np.ndarray = rng.normal(
#         loc=0.25, scale=0.25, size=(channel_count, image_size, image_size)
#     )
#     output_image: np.ndarray = np.zeros_like(a=image)

#     for channel_idx in range(channel_count):
#         channel: np.ndarray = image[channel_idx]

#         for _ in range(spot_count):
#             rr, cc = draw.disk(
#                 center=(rng.integers(channel.shape[0]), rng.integers(channel.shape[1])),
#                 radius=spot_radius,
#                 shape=channel.shape,
#             )
#             channel[rr, cc] = 1

#         channel *= rng.normal(loc=1.0, scale=0.1, size=channel.shape)

#         channel *= ndimage.zoom(
#             rng.normal(loc=1.0, scale=0.5, size=(cloud_noise_size, cloud_noise_size)),
#             image_size / cloud_noise_size,
#         )

#         output_image[channel_idx]: np.ndarray = ndimage.gaussian_filter(
#             channel, sigma=2.0
#         )

#     # Make temporary path
#     tmp_path: pathlib.Path = tmpdir_factory.mktemp("data")

#     for idx, output_channel in enumerate(output_image):
#         image_utils.save_image(fname=tmp_path / f"chan_{idx}.tiff", data=output_channel)

#     yield tmp_path


@pytest.fixture(scope="module")
def tiff_dir(tmpdir_factory: pytest.TempPathFactory) -> pathlib.Path:
    tiff_dir_name: pathlib.Path = tmpdir_factory.mktemp("tiff_dir")
    num_fovs: int = 3

    for nf in range(num_fovs):
        os.mkdir(tiff_dir_name / f"fov{nf}")
        tiff_file: pathlib.Path = tiff_dir_name / f"fov{nf}" / f"fov{nf}.tiff"
        tiff_data: np.ndarray = np.random.rand(32, 32)
        io.imsave(str(tiff_file), tiff_data)

    yield tiff_dir_name


@pytest.fixture(scope="module")
def seg_dir(tmpdir_factory: pytest.TempPathFactory) -> pathlib.Path:
    seg_dir_name: pathlib.Path = tmpdir_factory.mktemp("seg_dir")
    seg_subdirs: List[str] = ["seg1", "seg2"]
    num_fovs: int = 3

    for ss in seg_subdirs:
        os.mkdir(seg_dir_name / ss)

        for nf in range(num_fovs):
            mask_file: pathlib.Path = seg_dir_name / ss / f"fov{nf}_{ss}.tiff"
            mask_data: np.ndarray = np.random.randint(0, 2, (32, 32))
            io.imsave(str(mask_file), mask_data)

    yield seg_dir_name


@pytest.fixture(scope="module")
def mantis_dir(tmpdir_factory: pytest.TempPathFactory) -> pathlib.Path:
    mantis_dir_name: pathlib.Path = tmpdir_factory.mktemp("mantis_dir")
    yield mantis_dir_name


@pytest.fixture(scope="module")
def mask_dir(tmpdir_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Creates an example mask directory

    Yields:
        pathlib.Path:
            The path to the mask directory
    """

    mask_dir_name: pathlib.Path = tmpdir_factory.mktemp("mask_dir")

    mask_suffixes: List[str] = ["type1.tiff", "type2.tiff"]
    fov_count: int = 3
    cluster_count: int = 2

    mask_files: List[pathlib.Path] = [
        mask_dir_name / f"fov{fov_num}_{ms}"
        for fov_num in range(fov_count)
        for ms in mask_suffixes
    ]

    for mf in mask_files:
        mask_data: np.ndarray = np.random.randint(0, 3, (32, 32))
        io.imsave(str(mf), mask_data)

    yield mask_dir_name


def test_renumber_masks(mask_dir: pathlib.Path):
    ez_seg_utils.renumber_masks(mask_dir)
    mask_files = io_utils.list_files(mask_dir, substrs=".tiff")

    cluster_start: int = len(mask_files) * 2 + 1
    max_cluster: int = cluster_start + len(mask_files) * 2 - 1
    all_clusters_seen: np.ndarray = np.array([])

    for i, mf in enumerate(mask_files):
        mask_data: np.ndarray = io.imread(str(mask_dir / mf))
        mask_clusters: np.ndarray = np.unique(mask_data[mask_data > 0])
        all_clusters_seen = np.concatenate([all_clusters_seen, mask_clusters])

    assert np.all(np.sort(all_clusters_seen) == np.arange(cluster_start, max_cluster + 1))


def test_create_mantis_project(tiff_dir: pathlib.Path, seg_dir: pathlib.Path,
                               mantis_dir: pathlib.Path):
    fovs: List[str] = [f"fov{f}" for f in range(3)]
    ez_seg_utils.create_mantis_project(
        fovs,
        tiff_dir,
        seg_dir,
        mantis_dir
    )

    for fov in fovs:
        expected_files: List[str] = []
        expected_files.extend(io_utils.list_files(tiff_dir / fov, substrs=".tiff"))
        for seg_subdir in io_utils.list_folders(seg_dir):
            expected_files.extend(io_utils.list_files(seg_dir / seg_subdir, substrs=fov))

        mantis_files: List[str] = io_utils.list_files(mantis_dir / fov)
        assert set(expected_files) == set(mantis_files)


def test_log_creator():
    with tempfile.TemporaryDirectory() as td:
        log_dir: Union[str, pathlib.Path] = os.path.join(td, "log_dir")
        os.mkdir(log_dir)

        variables_to_log: dict[str, any] = {"var1": "val1", "var2": 2}
        ez_seg_utils.log_creator(variables_to_log, log_dir)

        with open(os.path.join(log_dir, "config_values.txt"), "r") as infile:
            log_lines: List[str] = infile.readlines()

        assert log_lines[0] == "var1: val1\n"
        assert log_lines[1] == "var2: 2\n"


def test_split_csvs_by_mask():
    pass
