import itertools
import math
import os
import pathlib
import random
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import skimage.io as io
from alpineer import io_utils, test_utils, misc_utils
from pytest_mock import MockerFixture

import ark.settings as settings
from ark.segmentation import fiber_segmentation


@pytest.fixture(scope="function")
def fiber_seg_data(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """Creates a set of test fovs and channels for running the fiber segmentation pipeline.

    Args:
        tmp_path (pathlib.Path): A temporary directory for storing the test data.

    Yields:
        Generator[pathlib.Path, None, None]: The path to the temporary directory.
    """

    data_dir: pathlib.Path = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    test_utils.create_paired_xarray_fovs(base_dir=data_dir, fov_names=["fov1", "fov2"],
                                         channel_names=["Collagen1", "chan0"],
                                         fills=False,
                                         img_shape=(200, 200))
    yield data_dir


def test_plot_fiber_segmentation_steps(fiber_seg_data: pathlib.Path):
    img_dir: pathlib.path = fiber_seg_data

    # bad directory should raise an errors
    with pytest.raises(FileNotFoundError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps('bad_dir', 'fov1', 'Collagen1')

    # bad channel should raise an errors
    with pytest.raises(ValueError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'bad_channel')

    # bad subdirectory should raise an errors
    with pytest.raises(FileNotFoundError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps(
            img_dir, 'fov1', 'Collagen1', img_sub_folder='bad_subdir')

    # test success
    fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'Collagen1')


def test_run_fiber_segmentation(fiber_seg_data: pathlib.Path, tmp_path: pathlib.Path,
                                mocker: MockerFixture):
    img_dir: pathlib.Path = fiber_seg_data
    out_dir: pathlib.Path = tmp_path / "fiber_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # bad directories should raise an error
    with pytest.raises(FileNotFoundError):
        _ = fiber_segmentation.run_fiber_segmentation('bad_path', 'Collagen1', out_dir)

    with pytest.raises(FileNotFoundError):
        _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', 'bad_path')

    # bad subdirectory should raise an errors
    with pytest.raises(FileNotFoundError):
        _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', out_dir,
                                                      img_sub_folder='bad_folder')

    # bad channel should raise an errors
    with pytest.raises(ValueError):
        _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'bad_channel', out_dir)

    # test success
    fiber_object_table = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', out_dir)

    # check all fovs are processed
    assert fiber_object_table[settings.FOV_ID].unique().sort() == \
        io_utils.list_folders(img_dir).sort()
    # check for fiber alignment column
    assert 'alignment_score' in fiber_object_table.columns

    # check output files
    for fov in io_utils.list_files(img_dir):
        assert os.path.exists(os.path.join(out_dir, f'{fov}_fiber_labels.tiff'))
    assert os.path.exists(os.path.join(out_dir, 'fiber_object_table.csv'))

    # test success with debugging
    fiber_object_table = fiber_segmentation.run_fiber_segmentation(
        img_dir, 'Collagen1', out_dir, debug=True)

    # check debug output files
    intermediate_imgs = ['fov1_thresholded.tiff', 'fov1_ridges_thresholded.tiff',
                         'fov1_frangi_filter.tiff', 'fov1_contrast_adjusted.tiff']
    for img in intermediate_imgs:
        img_path = os.path.join(out_dir, '_debug', img)
        assert os.path.exists(img_path)

    # check empty fiber images don't throw error
    mocker.patch('ark.segmentation.fiber_segmentation.segment_fibers', return_value=pd.DataFrame())
    empty_table = fiber_segmentation.run_fiber_segmentation(
        img_dir, 'Collagen1', out_dir, debug=True)


@pytest.mark.parametrize("neighbors", [1, 2])
def test_calculate_fiber_alignment(neighbors):
    ex_fiber_table = pd.DataFrame({
        'fov': ['fov1', 'fov1', 'fov1', 'fov1'],
        'label': [1, 2, 3, 4],
        'orientation': [-30, -15, 15, 0],
        'centroid-0': [0, 3, 1, 2],
        'centroid-1': [0, 3, 3, 2],
        'major_axis_length': [2, 2, 2, 1.5],
        'minor_axis_length': [1, 1, 1, 1]
    })
    closest_fibers = {
        1: [3, 2],
        2: [3, 1],
        3: [2, 1]
    }

    # calculate alignment scores with k=1
    align_table = fiber_segmentation.calculate_fiber_alignment(ex_fiber_table, k=neighbors)

    # check that fiber 4 was excluded
    short_fiber = align_table[align_table.label == 4]
    assert math.isnan(short_fiber.alignment_score)

    # check that alignments are calculated correctly, ignoring fiber 4 when it is closest
    for fiber in [1, 2, 3]:
        angle = int(ex_fiber_table[ex_fiber_table.label == fiber].orientation)
        neighbor_fibers = closest_fibers[fiber][:neighbors]
        neighbor_orientations = np.array(ex_fiber_table[ex_fiber_table.label.isin(neighbor_fibers)]
                                         .orientation.values)
        alignment_score = (np.sqrt(np.sum((neighbor_orientations - angle) ** 2)) / neighbors)

        assert (alignment_score == align_table[align_table.label == fiber].alignment_score).all()


@pytest.mark.parametrize(
    "areas", [random.sample(range(100, 200), 10), random.sample(range(100, 200), 10)])
def test_calculate_density(areas):
    ex_fiber_table = pd.DataFrame({
        'fov': ['fov1'] * len(areas),
        'label': list(range(1, 11)),
        'area': areas,
    })

    pixel_density, fiber_density = fiber_segmentation.calculate_density(ex_fiber_table,
                                                                        total_pixels=50**2)
    assert pixel_density == (np.sum(areas) / 50**2) * 100
    assert fiber_density == (len(areas) / 50**2) * 100


@pytest.mark.parametrize("min_fiber_num", [1, 5])
def test_generate_tile_stats(min_fiber_num):
    fov_length = 16
    fov_fiber_img = np.zeros((fov_length, fov_length))

    fov_fiber_table = pd.DataFrame({
        'fov': ['fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1'],
        'label': [1, 2, 3, 4, 5, 6],
        'centroid-0': [0, 1, 1, 0, 2, 9],
        'centroid-1': [0, 1, 0, 1, 2, 9],
        'major_axis_length': random.sample(range(1, 20), 6),
        'minor_axis_length': random.sample(range(1, 20), 6),
        'orientation': [random.uniform(-1.57, 1.57) for _ in range(6)],
        'area': [1] * 6,
        'eccentricity': [random.uniform(0, 1) for _ in range(6)],
        'euler_number': [random.choice([0, 1]) for _ in range(6)],
        'alignment_score': random.sample(range(10, 40), 6),
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # test success
        tile_length = 8
        tile_stats = fiber_segmentation.generate_tile_stats(
            fov_fiber_table, fov_fiber_img, fov_length, tile_length, min_fiber_num,
            temp_dir, save_tiles=True)

        # check tile level values
        # 0,0 tile, fov1 should exclude fiber 6 since located in different tile
        tile_0_0 = tile_stats[np.logical_and(tile_stats.tile_y == 0, tile_stats.tile_x == 0)]
        assert tile_0_0.avg_major_axis_length[0] \
            == np.mean(fov_fiber_table.major_axis_length[0:5])
        assert tile_0_0.avg_alignment_score[0] \
            == np.mean(fov_fiber_table.alignment_score[0:5])

        tile_8_8 = tile_stats[np.logical_and(tile_stats.tile_y == 8, tile_stats.tile_x == 8)]
        # make sure tile 8,8 has nan since there's only 1 fiber (5 fibers required for stat calc)
        if min_fiber_num == 5:
            assert math.isnan(tile_8_8.avg_major_axis_length)
            assert math.isnan(tile_8_8.avg_alignment_score)
        # check the correct value (1 fiber required for stat calc)
        elif min_fiber_num == 1:
            assert tile_8_8.avg_major_axis_length[3] == fov_fiber_table.major_axis_length[5]
            assert tile_8_8.avg_alignment_score[3] == fov_fiber_table.alignment_score[5]

        # check for saved tile images
        tile_corners = [tile_length * i for i in range(int(fov_length / tile_length))]
        for x, y, fov in \
                itertools.product(tile_corners, tile_corners, np.unique(fov_fiber_table.fov)):
            assert os.path.exists(os.path.join(temp_dir, fov, f'tile_{y},{x}.tiff'))

        # check additional property averages are included
        misc_utils.verify_in_list(
            summary_stats=["avg_major_axis_length", "avg_minor_axis_length", "avg_orientation",
                           "avg_area", "avg_eccentricity", "avg_euler_number"],
            output_columns=tile_stats.columns)


@pytest.mark.parametrize("min_fiber_num", [1, 5])
def test_generate_summary_stats(mocker: MockerFixture, min_fiber_num):
    fov_length = 16
    mocker.patch('skimage.io.imread', return_value=np.zeros((fov_length, fov_length)))

    fiber_object_table = pd.DataFrame({
        'fov': ['fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1',
                'fov2', 'fov2', 'fov2', 'fov2', 'fov2', 'fov2'],
        'label': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        'centroid-0': random.sample(range(0, 15), 12),
        'centroid-1': random.sample(range(0, 15), 12),
        'major_axis_length': random.sample(range(1, 20), 12),
        'minor_axis_length': random.sample(range(1, 20), 12),
        'orientation': [random.uniform(-1.57, 1.57) for _ in range(12)],
        'area': [1]*12,
        'eccentricity': [random.uniform(0, 1) for _ in range(12)],
        'euler_number': [random.choice([0, 1]) for _ in range(12)],
        'alignment_score': random.sample(range(10, 40), 12),
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad tile size should raise an error
        with pytest.raises(ValueError, match="Tile length must be a factor"):
            _, _ = fiber_segmentation.generate_summary_stats(None, temp_dir, tile_length=5)

        # test success
        tile_length = 8
        fov_stats, tile_stats = fiber_segmentation.generate_summary_stats(
            fiber_object_table, temp_dir, tile_length=tile_length, save_tiles=True)

        assert os.path.exists(os.path.join(temp_dir, 'fiber_stats_table.csv'))
        tile_dir = os.path.join(temp_dir, f'tile_stats_{tile_length}')
        assert os.path.exists(os.path.join(temp_dir, tile_dir,
                                           f'fiber_stats_table-tile_{tile_length}.csv'))

        # check fov-level values
        # only confirm fiber property avg stats, densities are tested above
        misc_utils.verify_in_list(
            summary_stats=["avg_major_axis_length", "avg_minor_axis_length", "avg_orientation",
                           "avg_area", "avg_eccentricity", "avg_euler_number",
                           "avg_alignment_score"], output_columns=fov_stats.columns)
        assert fov_stats.avg_major_axis_length[0] ==\
            np.mean(fiber_object_table.major_axis_length[0:6])
        assert fov_stats.avg_major_axis_length[1] == \
            np.mean(fiber_object_table.major_axis_length[6:12])
