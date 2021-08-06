import copy
import json
import numpy as np
import os
import pytest
import random
import tempfile

import ark.mibi.tiling_utils as tiling_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def test_set_tiling_params(monkeypatch):
    # define a sample set of fovs
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # let's just set all the user inputs to 1 to make the test easy
    user_input = 1

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: 1)

    # bad fov list path provided
    with pytest.raises(FileNotFoundError):
        tiling_utils.set_tiling_params('bad_fov_list_path.json', 'bad_moly_path.json')

    with tempfile.TemporaryDirectory() as temp_dir:
        # write fov list
        sample_fov_list_path = os.path.join(temp_dir, 'fov_list.json')
        with open(sample_fov_list_path, 'w') as fl:
            json.dump(sample_fovs_list, fl)

        # bad moly path provided
        with pytest.raises(FileNotFoundError):
            tiling_utils.set_tiling_params(sample_fov_list_path, 'bad_moly_path.json')

        # write moly point
        sample_moly_path = os.path.join(temp_dir, 'moly_point.json')
        with open(sample_moly_path, 'w') as moly:
            json.dump(sample_moly_point, moly)

        # run tiling parameter setting process with predefined user inputs
        sample_tiling_params, moly_point = tiling_utils.set_tiling_params(
            sample_fov_list_path, sample_moly_path
        )

        # assert the fovs in the tiling params are the same as in the original fovs list
        assert sample_tiling_params['fovs'] == sample_fovs_list['fovs']

        # assert region start x and region start y values are correct
        region_start_x = sample_tiling_params['region_start_x']
        region_start_y = sample_tiling_params['region_start_y']
        fov_0 = sample_fovs_list['fovs'][0]
        fov_1 = sample_fovs_list['fovs'][1]

        assert region_start_x[0] == fov_0['centerPointMicrons']['x']
        assert region_start_x[1] == fov_1['centerPointMicrons']['x']
        assert region_start_y[0] == fov_0['centerPointMicrons']['y']
        assert region_start_y[1] == fov_1['centerPointMicrons']['y']

        # assert fov_num_x and fov_num_y are all set to 1
        assert all(nx == 1 for nx in sample_tiling_params['fov_num_x'])
        assert all(ny == 1 for ny in sample_tiling_params['fov_num_y'])

        # assert x_fov_size and y_fov_size are all set to 1
        assert all(sx == 1 for sx in sample_tiling_params['x_fov_size'])
        assert all(sy == 1 for sy in sample_tiling_params['y_fov_size'])

        # assert randomize is set to 1 for both fovs
        assert all(r == 1 for r in sample_tiling_params['randomize'])

        # assert moly run is set to 1
        assert sample_tiling_params['moly_run'] == 1

        # assert moly interval is set to 1
        assert sample_tiling_params['moly_interval'] == 1


def test_create_tiled_regions():
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    sample_tiling_params = {
        'fovFormatVersion': '1.5',
        'fovs': sample_fovs_list['fovs'],
        'region_start_x': [0, 50],
        'region_start_y': [100, 150],
        'fov_num_x': [2, 4],
        'fov_num_y': [4, 2],
        'x_fov_size': [5, 10],
        'y_fov_size': [10, 5],
        'moly_run': 0,
        'moly_interval': 3
    }

    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # we need these globals to set baselines for both moly interval settings
    actual_center_points_no_run_no_int = None
    actual_center_points_no_run_int = None
    actual_center_points_run_no_int = None
    actual_center_points_run_int = None

    # test randomization for no fovs, some fovs, and all fovs
    for randomize_setting in [[0, 0], [0, 1], [1, 1]]:
        sample_tiling_params['randomize'] = randomize_setting

        for moly_run in [0, 1]:
            sample_tiling_params['moly_run'] = moly_run

            for moly_interval_setting in [False, True]:
                # place the moly_interval param in if True, else remove
                if moly_interval_setting:
                    sample_tiling_params['moly_interval'] = 3
                else:
                    del sample_tiling_params['moly_interval']

                # create the tiles
                tiled_regions = tiling_utils.create_tiled_regions(
                    sample_tiling_params, sample_moly_point
                )

                # retrieve the center points
                center_points = [
                    (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
                    for fov in tiled_regions['fovs']
                ]

                if randomize_setting == [0, 0] and moly_run:
                    if moly_interval_setting:
                        # since True will run after False for moly_interval_setting
                        # actual_center_points_no_run_no_int will be set so we can piggyback
                        actual_center_points_run_int = actual_center_points_run_no_int[:]

                        # define the indices to add moly points
                        moly_indices_run = [3, 7, 12, 16, 20]

                        # insert moly points
                        for mi in moly_indices_run:
                            actual_center_points_run_int.insert(mi, (14540, -10830))

                        # the center points should match up exactly with no sorting
                        assert center_points == actual_center_points_run_int
                    else:
                        # set a baseline for what the center points should be
                        actual_center_points_run_no_int = [
                            (x, y) for x in np.arange(0, 10, 5) for y in np.arange(100, 140, 10)
                        ] + [(14540, -10830)] + [
                            (x, y) for x in np.arange(50, 90, 10) for y in np.arange(150, 160, 5)
                        ]

                        # the center points should match up exactly with no sorting
                        assert center_points == actual_center_points_run_no_int
                elif randomize_setting == [0, 0] and not moly_run:
                    if moly_interval_setting:
                        # since True will run after False for moly_interval_setting
                        # actual_center_points_no_run_no_int will be set so we can piggyback
                        actual_center_points_no_run_int = actual_center_points_no_run_no_int[:]

                        # define the indices to add moly points
                        moly_indices_no_run = [3, 7, 11, 15, 19]

                        # insert moly points
                        for mi in moly_indices_no_run:
                            actual_center_points_no_run_int.insert(mi, (14540, -10830))

                        # the center points should match up exactly with no sorting
                        assert center_points == actual_center_points_no_run_int
                    else:
                        # set a baseline for what the center points should be
                        actual_center_points_no_run_no_int = [
                            (x, y) for x in np.arange(0, 10, 5) for y in np.arange(100, 140, 10)
                        ] + [
                            (x, y) for x in np.arange(50, 90, 10) for y in np.arange(150, 160, 5)
                        ]

                        # the center points should match up exactly with no sorting
                        assert center_points == actual_center_points_no_run_no_int

                elif randomize_setting == [0, 1]:
                    if moly_run:
                        # define the end of fov 1
                        fov_1_end = 11 if moly_interval_setting else 9

                        # define the baseline points we will be testing against
                        actual_points = actual_center_points_run_int if moly_interval_setting \
                            else actual_center_points_run_no_int
                    else:
                        # define the end of fov 1
                        fov_1_end = 10 if moly_interval_setting else 8

                        # define the baseline points we will be testing against
                        actual_points = actual_center_points_no_run_int if moly_interval_setting \
                            else actual_center_points_no_run_no_int

                    # ensure the fov 1 center points are the same for both sorted and random
                    assert center_points[:fov_1_end] == actual_points[:fov_1_end]

                    # ensure the random center points for fov 2 contain the same elements
                    # as its sorted version
                    misc_utils.verify_same_elements(
                        computed_center_points=center_points[fov_1_end:],
                        actual_center_points=actual_points[fov_1_end:]
                    )

                    # however, fov 2 sorted entries should NOT equal fov 2 random entries
                    # NOTE: due to randomization, this test will fail once in a blue moon
                    assert center_points[fov_1_end:] != actual_points[fov_1_end:]

                elif randomize_setting == [1, 1]:
                    if moly_run:
                        # define the end of fov 1
                        fov_1_end = 11 if moly_interval_setting else 9

                        # define the baseline points we will be testing against
                        actual_points = actual_center_points_run_int if moly_interval_setting \
                            else actual_center_points_run_no_int
                    else:
                        # define the end of fov 1
                        fov_1_end = 10 if moly_interval_setting else 8

                        # define the baseline points we will be testing against
                        actual_points = actual_center_points_no_run_int if moly_interval_setting \
                            else actual_center_points_no_run_no_int

                    # ensure the random center points for fov 1 contain the same elements
                    # as its sorted version
                    misc_utils.verify_same_elements(
                        computed_center_points=center_points[:fov_1_end],
                        actual_center_points=actual_points[:fov_1_end]
                    )

                    # however, fov 1 sorted entries should NOT equal fov 1 random entries
                    # NOTE: due to randomization, this test will fail once in a blue moon
                    assert center_points[:fov_1_end] != actual_points[:fov_1_end]

                    # ensure the random center points for fov 2 contain the same elements
                    # as its sorted version
                    misc_utils.verify_same_elements(
                        computed_center_points=center_points[fov_1_end:],
                        actual_center_points=actual_points[fov_1_end:]
                    )

                    # however, fov 2 sorted entries should NOT equal fov 2 random entries
                    # NOTE: due to randomization, this test will fail once in a blue moon
                    assert center_points[fov_1_end:] != actual_points[fov_1_end:]
