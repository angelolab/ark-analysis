import copy
import json
import os
import pytest
import random
import tempfile

import ark.mibi.tiling_utils as tiling_utils
import ark.utils.misc_utils as misc_utils


def test_set_tiling_params(monkeypatch):
    # define a sample set of fovs
    sample_fovs_list = {
        "exportDateTime": "2021-03-12T19:02:37.920Z",
        "fovFormatVersion": "1.5",
        "fovs": [
            {
              "scanCount": 1,
              "centerPointMicrons": {
                "x": 8758,
                "y": 38150
              },
              "timingChoice": 7,
              "frameSizePixels": {
                "width": 2048,
                "height": 2048
              },
              "imagingPreset": {
                "preset": "Normal",
                "aperture": "2",
                "displayName": "Fine",
                "defaults": {
                  "timingChoice": 7
                }
              },
              "sectionId": 8201,
              "slideId": 5931,
              "name": "TheFirstFOV",
              "timingDescription": "1 ms"
            },
            {
              "scanCount": 1,
              "centerPointMicrons": {
                "x": 5142,
                "y": 6371
              },
              "timingChoice": 7,
              "frameSizePixels": {
                "width": 2048,
                "height": 2048
              },
              "imagingPreset": {
                "preset": "Normal",
                "aperture": "2",
                "displayName": "Fine",
                "defaults": {
                  "timingChoice": 7
                }
              },
              "sectionId": 8201,
              "slideId": 5931,
              "name": "TheSecondFOV",
              "timingDescription": "1 ms"
            }
        ]
    }

    # set moly point
    sample_moly_point = {
        "scanCount": 3,
        "centerPointMicrons": {
            "x": 14540,
            "y": -10830
        },
        "fovSizeMicrons": 200,
        "timingChoice": 7,
        "frameSizePixels": {
            "width": 128,
            "height": 128
        },
        "imagingPreset": {
            "preset": "Tuning",
            "aperture": "3",
            "displayName": "QC - 100µm",
            "defaults": {
                "timingChoice": 7
            }
        },
        "standardTarget": "Moly Foil",
        "name": "MoQC",
        "notes": None,
        "timingDescription": "1 ms"
    }

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
    # set tiling_params dict
    sample_tiling_params = {
        'fovFormatVersion': '1.5',
        'fovs': [
            {
                'scanCount': 1,
                'centerPointMicrons': {
                    'x': 0, 'y': 0
                },
                'timingChoice': 7,
                'frameSizePixels': {
                    'width': 2048, 'height': 2048
                },
                'imagingPreset': {
                    'preset': 'Normal',
                    'aperture': '2',
                    'displayName': 'Fine',
                    'defaults': {
                        'timingChoice': 7
                    }
                },
                'sectionId': 8201,
                'slideId': 5931,
                'name': 'TheFirstFOV',
                'timingDescription': '1 ms'
            },
            {
                'scanCount': 1,
                'centerPointMicrons': {
                    'x': 100, 'y': 100
                },
                'timingChoice': 7,
                'frameSizePixels': {
                    'width': 2048, 'height': 2048
                },
                'imagingPreset': {
                    'preset': 'Normal',
                    'aperture': '2',
                    'displayName': 'Fine',
                    'defaults': {
                        'timingChoice': 7
                    }
                },
                'sectionId': 8201,
                'slideId': 5931,
                'name': 'TheSecondFOV',
                'timingDescription': '1 ms'
            }
        ],
        'region_start_x': [0, 50],
        'region_start_y': [100, 150],
        'fov_num_x': [2, 4],
        'fov_num_y': [4, 2],
        'x_fov_size': [5, 10],
        'y_fov_size': [10, 5],
        'randomize': [0, 0],
        'moly_run': 1
    }

    # set moly point
    sample_moly_point = {
        "scanCount": 3,
        "centerPointMicrons": {
            "x": 14540,
            "y": -10830
        },
        "fovSizeMicrons": 200,
        "timingChoice": 7,
        "frameSizePixels": {
            "width": 128,
            "height": 128
        },
        "imagingPreset": {
            "preset": "Tuning",
            "aperture": "3",
            "displayName": "QC - 100µm",
            "defaults": {
                "timingChoice": 7
            }
        },
        "standardTarget": "Moly Foil",
        "name": "MoQC",
        "notes": None,
        "timingDescription": "1 ms"
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # test 1: no randomization, no additional moly point interval, moly points between runs
        tiled_regions_base = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        sorted_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_base['fovs']
        ]

        # define the actual center points in the same expected order
        actual_center_points_run = [
            (0, 100), (0, 110), (0, 120), (0, 130), (5, 100),
            (5, 110), (5, 120), (5, 130), (14540, -10830),
            (50, 150), (50, 155), (60, 150), (60, 155), (70, 150),
            (70, 155), (80, 150), (80, 155)
        ]

        # check that the created center points equal the sorted center points
        assert sorted_center_points_run == actual_center_points_run

        # test 2: no randomization, no additional moly point interval, no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        sorted_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_no_run['fovs']
        ]

        # define the actual center points in the same expected order
        actual_center_points_no_run = actual_center_points_run[:]
        actual_center_points_no_run.remove((14540, -10830))

        # check that the created center points equal the sorted center points
        assert sorted_center_points_no_run == actual_center_points_no_run

        # test 3: randomization for one fov, no additional moly point interval,
        # moly points between runs
        sample_tiling_params['randomize'] = [0, 1]
        sample_tiling_params['moly_run'] = 1
        tiled_regions_random_some_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_some_run['fovs']
        ]

        # ensure the moly point is inserted in the same spot as if the fovs were sorted
        random_center_points_run[8] == sorted_center_points_run[8]

        # ensure the fov 1 center points are the same for both sorted and random
        assert random_center_points_run[:8] == sorted_center_points_run[:8]

        # ensure the random center points for fov 2 contain the same elements as its sorted version
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_run[9:],
            sorted_center_points=sorted_center_points_run[9:]
        )

        # the random center points for fov 2 should be ordered differently than its sorted version
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_run[9:] != sorted_center_points_run[9:]

        # test 4: randomization for one fov, no additional moly point interval,
        # no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_random_some_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_some_no_run['fovs']
        ]

        # ensure that fov 1 center points are the same for both sorted and random
        assert random_center_points_no_run[:8] == sorted_center_points_no_run[:8]

        # ensure the random center points for fov 2 contain the same elements as its sorted version
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_run[8:],
            sorted_center_points=sorted_center_points_run[8:]
        )

        # the random center points for fov 2 should be ordered differently than its sorted version
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_run[8:] != sorted_center_points_run[8:]

        # test 5: randomization for both fovs, no additional moly point interval,
        # moly points between runs
        sample_tiling_params['randomize'] = [1, 1]
        sample_tiling_params['moly_run'] = 1
        tiled_regions_random_all_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_all_run['fovs']
        ]

        # ensure the moly point is inserted in the same spot as if the fovs were sorted
        assert random_center_points_run[8] == sorted_center_points_run[8]

        # make sure the random center points contain the same elements as the sorted center points
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_run,
            sorted_center_points=sorted_center_points_run
        )

        # test 6: randomization for both fovs, no additional moly point interval,
        # no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_random_all_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_all_no_run['fovs']
        ]

        # make sure the random center points contain the same elements as the sorted center points
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_no_run,
            sorted_center_points=sorted_center_points_no_run
        )

        # the random center points should be ordered differently than the sorted center points
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_no_run != sorted_center_points_no_run

        # test 7: no randomization, additional moly point interval, moly points between runs
        sample_tiling_params['randomize'] = [0, 0]
        sample_tiling_params['moly_run'] = 1
        sample_tiling_params['moly_interval'] = 3
        tiled_regions_moly_int_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        sorted_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_moly_int_run['fovs']
        ]

        # define the actual center points in the same expected order with additional moly point
        actual_center_points_run = [
            (0, 100), (0, 110), (0, 120), (14540, -10830),
            (0, 130), (5, 100), (5, 110), (14540, -10830),
            (5, 120), (5, 130), (14540, -10830), (50, 150),
            (14540, -10830), (50, 155), (60, 150), (60, 155),
            (14540, -10830), (70, 150), (70, 155), (80, 150),
            (14540, -10830), (80, 155)
        ]

        # check that the created center points equal the sorted center points
        assert sorted_center_points_run == actual_center_points_run

        # test 8: no randomization, additional moly point interval, no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_moly_int_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        sorted_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_moly_int_no_run['fovs']
        ]

        # define the actual center points in the same expected order with additional moly point
        actual_center_points_no_run = actual_center_points_run[:]
        del actual_center_points_no_run[10]

        assert sorted_center_points_no_run == actual_center_points_no_run

        # test 9: randomization for one fov, additional moly point interval,
        # moly points between runs
        sample_tiling_params['randomize'] = [0, 1]
        sample_tiling_params['moly_run'] = 1
        tiled_regions_random_some_moly_int_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_some_moly_int_run['fovs']
        ]

        # define the moly point indices, applies for both sorted and random tiled regions
        moly_indices_run = [3, 6, 8, 10, 13, 16]

        # assert that each moly index is the same in both sorted and random tiled regions
        for mi in moly_indices_run:
            random_center_points_run[mi] == sorted_center_points_run[mi]

        # ensure the fov 1 center points are the same for both sorted and random
        assert random_center_points_run[:10] == sorted_center_points_run[:10]

        # ensure the random center points for fov 2 contain the same elements as its sorted version
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_run[11:],
            sorted_center_points=sorted_center_points_run[11:]
        )

        # the random center points for fov 2 should be ordered differently than its sorted version
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_run[11:] != sorted_center_points_run[11:]

        # test 10: randomization for one fov, additional moly point interval,
        # no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_random_some_moly_int_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_some_moly_int_no_run['fovs']
        ]

        # define the moly point indices, applies for both sorted and random tiled regions
        moly_indices_no_run = [3, 6, 8, 12, 15]

        # assert that each moly index is the same in both sorted and random tiled regions
        for mi in moly_indices_run:
            random_center_points_run[mi] == sorted_center_points_run[mi]

        # ensure the fov 1 center points are the same for both sorted and random
        assert random_center_points_no_run[:10] == sorted_center_points_no_run[:10]

        # ensure the random center points for fov 2 contain the same elements as its sorted version
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_no_run[10:],
            sorted_center_points=sorted_center_points_no_run[10:]
        )

        # test 11: randomization for both fovs, additional moly point interval,
        # moly points between runs
        sample_tiling_params['randomize'] = [1, 1]
        sample_tiling_params['moly_run'] = 1
        tiled_regions_random_all_moly_int_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_all_moly_int_run['fovs']
        ]

        # assert that each moly index is the same in both sorted and random tiled regions
        for mi in moly_indices_run:
            random_center_points_run[mi] == sorted_center_points_run[mi]

        # make sure the random center points contain the same elements as the sorted center points
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_run,
            sorted_center_points=sorted_center_points_run
        )

        # the random center points should be ordered differently than the sorted center points
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_run != sorted_center_points_run

        # test 12: randomization for both fovs, additional moly point interval,
        # no moly points between runs
        sample_tiling_params['moly_run'] = 0
        tiled_regions_random_all_moly_int_no_run = tiling_utils.create_tiled_regions(
            sample_tiling_params, sample_moly_point
        )

        # get the center points created
        random_center_points_no_run = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in tiled_regions_random_all_moly_int_no_run['fovs']
        ]

        # assert that each moly index is the same in both sorted and random tiled regions
        for mi in moly_indices_no_run:
            random_center_points_no_run[mi] == sorted_center_points_no_run[mi]

        # make sure the random center points contain the same elements as the sorted center points
        misc_utils.verify_same_elements(
            random_center_points=random_center_points_no_run,
            sorted_center_points=sorted_center_points_no_run
        )

        # the random center points should be ordered differently than the sorted center points
        # due to the nature of randomization, this test will fail once in a blue moon
        assert random_center_points_no_run != sorted_center_points_no_run
