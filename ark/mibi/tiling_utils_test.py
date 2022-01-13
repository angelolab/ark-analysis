import copy
import json
import numpy as np
import os
import pytest
import random
import tempfile

from ark.mibi import tiling_utils
import ark.settings as settings
from ark.utils import misc_utils
from ark.utils import test_utils


# for script tiling
_TMA_TEST_CASES = [False, True]
_AUTO_RANDOMIZE_TEST_CASES = [['N', 'N'], ['N', 'Y'], ['Y', 'Y']]
_AUTO_MOLY_REGION_CASES = ['N', 'Y']
_AUTO_MOLY_INTERVAL_SETTING_CASES = [False, True]
_AUTO_MOLY_INTERVAL_VALUE_CASES = [3, 4]

# for remapping
_REMAP_RANDOMIZE_TEST_CASES = [False, True]
_REMAP_MOLY_INSERT_CASES = [False, True]
_REMAP_MOLY_INTERVAL_CASES = [4, 2]


def test_assign_metadata_vals():
    example_input_dict = {
        1: 'hello',
        2: False,
        3: 5.1,
        4: 7,
        5: {'do': 'not copy'},
        6: ['blah'],
        7: None
    }

    example_output_dict = {
        3: True,
        4: 1,
        5: {'hello': 'world'},
        6: 3.14,
        7: None
    }

    example_keys_ignore = [2, 4, 6]

    # tests a few things
    # 1. valid metadata keys are copied over from input_dict to output_dict
    # 2. keys_ignore do not make it into output_dict
    # 3. if a metadata key in input_dict exists in output_dict, it gets overwritten
    # 4. everything in output_dict that shouldn't get overwritten stays the same
    # 5. do not copy over non str, bool, int, or float values (ex. dict)
    # 6. if a value in keys_ignore doesn't exist in input_dict, ignore
    new_output_dict = tiling_utils.assign_metadata_vals(
        example_input_dict, example_output_dict, example_keys_ignore
    )

    # assert the keys are correct
    misc_utils.verify_same_elements(
        new_output_keys=list(new_output_dict.keys()),
        valid_keys=[1, 3, 4, 5, 6, 7]
    )

    # assert the values in each key is correct
    assert new_output_dict[3] == 5.1
    assert new_output_dict[4] == 1
    assert 'hello' in new_output_dict[5] and new_output_dict[5]['hello'] == 'world'
    assert new_output_dict[6] == 3.14
    assert new_output_dict[7] is None


def test_read_tiling_param(monkeypatch):
    # test 1: int inputs
    # test an incorrect non-int response, an incorrect int response, then a correct response
    user_inputs_int = iter(['N', 0, 1])

    # make sure the function receives the incorrect input first then the correct input
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs_int))

    # simulate the input sequence for
    sample_tiling_param = tiling_utils.read_tiling_param(
        "Sample prompt: ",
        "Sample error message",
        lambda x: x == 1,
        dtype=int
    )

    # assert sample_tiling_param was set to 1
    assert sample_tiling_param == 1

    # test 2: str inputs
    # test an incorrect non-str response, then an incorrect str response, then a correct response
    user_inputs_str = iter([1, 'N', 'Y'])

    # make sure the function receives the incorrect input first then the correct input
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs_str))

    # simulate the input sequence for
    sample_tiling_param = tiling_utils.read_tiling_param(
        "Sample prompt: ",
        "Sample error message",
        lambda x: x == 'Y',
        dtype=str
    )

    # assert sample_tiling_param was set to 'Y'
    assert sample_tiling_param == 'Y'


def test_tiled_region_read_input(monkeypatch):
    # define a sample fovs list
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    # define sample region_params to read data into
    sample_region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # set the user inputs
    user_inputs = iter([3, 3, 1, 1, 'Y', 3, 3, 1, 1, 'Y'])

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    # use the dummy user data to read values into the params lists
    tiling_utils.tiled_region_read_input(
        sample_fovs_list, sample_region_params
    )

    # assert the values were set properly
    assert sample_region_params['region_start_x'] == [0, 100]
    assert sample_region_params['region_start_y'] == [0, 100]
    assert sample_region_params['fov_num_x'] == [3, 3]
    assert sample_region_params['fov_num_y'] == [3, 3]
    assert sample_region_params['x_fov_size'] == [1, 1]
    assert sample_region_params['y_fov_size'] == [1, 1]
    assert sample_region_params['region_rand'] == ['Y', 'Y']


def test_generate_region_info():
    sample_region_inputs = {
        'region_start_x': [1, 1],
        'region_start_y': [2, 2],
        'fov_num_x': [3, 3],
        'fov_num_y': [4, 4],
        'x_fov_size': [5, 5],
        'y_fov_size': [6, 6],
        'region_rand': ['Y', 'Y']
    }

    # generate the region params
    sample_region_params = tiling_utils.generate_region_info(sample_region_inputs)

    # assert both region_start_x's are 1
    assert all(
        sample_region_params[i]['region_start_x'] == 1 for i in
        range(len(sample_region_params))
    )

    # assert both region_start_y's are 2
    assert all(
        sample_region_params[i]['region_start_y'] == 2 for i in
        range(len(sample_region_params))
    )

    # assert both num_fov_x's are 3
    assert all(
        sample_region_params[i]['fov_num_x'] == 3 for i in
        range(len(sample_region_params))
    )

    # assert both num_fov_y's are 4
    assert all(
        sample_region_params[i]['fov_num_y'] == 4 for i in
        range(len(sample_region_params))
    )

    # assert both x_fov_size's are 5
    assert all(
        sample_region_params[i]['x_fov_size'] == 5 for i in
        range(len(sample_region_params))
    )

    # assert both y_fov_size's are 6
    assert all(
        sample_region_params[i]['y_fov_size'] == 6 for i in
        range(len(sample_region_params))
    )

    # assert both randomize's are 0
    assert all(
        sample_region_params[i]['region_rand'] == 'Y' for i in
        range(len(sample_region_params))
    )


def test_tiled_region_set_params(monkeypatch):
    # define a sample set of fovs
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # set the user inputs
    user_inputs = iter([3, 3, 1, 1, 'Y', 3, 3, 1, 1, 'Y', 'Y', 'Y', 1])

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    # bad fov list path provided
    with pytest.raises(FileNotFoundError):
        tiling_utils.tiled_region_set_params('bad_fov_list_path.json', 'bad_moly_path.json')

    with tempfile.TemporaryDirectory() as temp_dir:
        # write fov list
        sample_fov_list_path = os.path.join(temp_dir, 'fov_list.json')
        with open(sample_fov_list_path, 'w') as fl:
            json.dump(sample_fovs_list, fl)

        # bad moly path provided
        with pytest.raises(FileNotFoundError):
            tiling_utils.tiled_region_set_params(sample_fov_list_path, 'bad_moly_path.json')

        # write moly point
        sample_moly_path = os.path.join(temp_dir, 'moly_point.json')
        with open(sample_moly_path, 'w') as moly:
            json.dump(sample_moly_point, moly)

        # run tiling parameter setting process with predefined user inputs
        sample_tiling_params, moly_point = tiling_utils.tiled_region_set_params(
            sample_fov_list_path, sample_moly_path
        )

        # assert the fovs in the tiling params are the same as in the original fovs list
        assert sample_tiling_params['fovs'] == sample_fovs_list['fovs']

        # assert region start x and region start y values are correct
        sample_region_params = sample_tiling_params['region_params']
        fov_0 = sample_fovs_list['fovs'][0]
        fov_1 = sample_fovs_list['fovs'][1]

        assert sample_region_params[0]['region_start_x'] == fov_0['centerPointMicrons']['x']
        assert sample_region_params[1]['region_start_x'] == fov_1['centerPointMicrons']['x']
        assert sample_region_params[0]['region_start_y'] == fov_0['centerPointMicrons']['y']
        assert sample_region_params[1]['region_start_y'] == fov_1['centerPointMicrons']['y']

        # assert fov_num_x and fov_num_y are all set to 3
        assert all(
            sample_region_params[i]['fov_num_x'] == 3 for i in
            range(len(sample_region_params))
        )
        assert all(
            sample_region_params[i]['fov_num_y'] == 3 for i in
            range(len(sample_region_params))
        )

        # assert x_fov_size and y_fov_size are all set to 1
        assert all(
            sample_region_params[i]['x_fov_size'] == 1 for i in
            range(len(sample_region_params))
        )
        assert all(
            sample_region_params[i]['y_fov_size'] == 1 for i in
            range(len(sample_region_params))
        )

        # assert randomize is set to Y for both fovs
        assert all(
            sample_region_params[i]['region_rand'] == 'Y' for i in
            range(len(sample_region_params))
        )

        # assert moly region is set to Y
        assert sample_tiling_params['moly_region'] == 'Y'

        # assert moly interval is set to 1
        assert sample_tiling_params['moly_interval'] == 1


def test_generate_x_y_fov_pairs():
    # define sample x and y pair lists
    sample_x_range = [0, 5]
    sample_y_range = [2, 4]

    # generate the sample (x, y) pairs
    sample_pairs = tiling_utils.generate_x_y_fov_pairs(sample_x_range, sample_y_range)

    assert sample_pairs == [(0, 2), (0, 4), (5, 2), (5, 4)]


@pytest.mark.parametrize('randomize_setting', _AUTO_RANDOMIZE_TEST_CASES)
@pytest.mark.parametrize('moly_region', _AUTO_MOLY_REGION_CASES)
@pytest.mark.parametrize('moly_interval_setting', _AUTO_MOLY_INTERVAL_SETTING_CASES)
@pytest.mark.parametrize('moly_interval_value', _AUTO_MOLY_INTERVAL_VALUE_CASES)
def test_tiled_region_generate_fov_list(randomize_setting, moly_region,
                                        moly_interval_setting, moly_interval_value):
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    sample_region_inputs = {
        'region_start_x': [0, 50],
        'region_start_y': [100, 150],
        'fov_num_x': [2, 4],
        'fov_num_y': [4, 2],
        'x_fov_size': [5, 10],
        'y_fov_size': [10, 5],
        'region_rand': ['N', 'N']
    }

    sample_region_params = tiling_utils.generate_region_info(sample_region_inputs)

    sample_tiling_params = {
        'fovFormatVersion': '1.5',
        'fovs': sample_fovs_list['fovs'],
        'region_params': sample_region_params
    }

    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    sample_tiling_params['moly_region'] = moly_region

    sample_tiling_params['region_params'][0]['region_rand'] = randomize_setting[0]
    sample_tiling_params['region_params'][1]['region_rand'] = randomize_setting[1]

    if moly_interval_setting:
        sample_tiling_params['moly_interval'] = moly_interval_value

    fov_regions = tiling_utils.tiled_region_generate_fov_list(
        sample_tiling_params, sample_moly_point
    )

    # assert none of the metadata keys explicitly added by set_tiling_params appear
    for k in ['region_params', 'moly_region', 'moly_interval']:
        assert k not in fov_regions

    # retrieve the center points
    center_points = [
        (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
        for fov in fov_regions['fovs']
    ]

    # define the center points sorted
    actual_center_points_sorted = [
        (x, y) for x in np.arange(0, 10, 5) for y in list(reversed(np.arange(70, 110, 10)))
    ] + [
        (x, y) for x in np.arange(50, 90, 10) for y in list(reversed(np.arange(145, 155, 5)))
    ]

    # if moly_region is Y, add a point in between the two regions
    if moly_region == 'Y':
        actual_center_points_sorted.insert(8, (14540, -10830))

    # add moly points in between if moly_interval_setting is set
    if moly_interval_setting:
        if moly_region == 'N':
            if moly_interval_value == 3:
                moly_indices = [3, 7, 11, 15, 19]
            else:
                moly_indices = [4, 13]
        else:
            if moly_interval_value == 3:
                moly_indices = [3, 7, 12, 16, 20]
            else:
                moly_indices = [4, 14]

        for mi in moly_indices:
            actual_center_points_sorted.insert(mi, (14540, -10830))

    # easiest case: the center points should be sorted
    if randomize_setting == ['N', 'N']:
        assert center_points == actual_center_points_sorted
    # if there's any sort of randomization involved
    else:
        # need to define the end of region 1
        if moly_region == 'N' and not moly_interval_setting:
            fov_1_end = 8
        elif moly_region == 'N' and moly_interval_setting:
            fov_1_end = 10 if moly_interval_value == 3 else 9
        elif moly_region == 'Y' and not moly_interval_setting:
            fov_1_end = 9
        elif moly_region and moly_interval_setting:
            fov_1_end = 11 if moly_interval_value == 3 else 10

        # only the second region is randomized
        if randomize_setting == ['N', 'Y']:
            # ensure the fov 1 center points are the same for both sorted and random
            assert center_points[:fov_1_end] == actual_center_points_sorted[:fov_1_end]

            # ensure the random center points for fov 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[fov_1_end:],
                actual_center_points=actual_center_points_sorted[fov_1_end:]
            )

            # however, fov 2 sorted entries should NOT equal fov 2 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[fov_1_end:] != actual_center_points_sorted[fov_1_end:]
        # both regions are randomized
        else:
            # ensure the random center points for fov 1 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[:fov_1_end],
                actual_center_points=actual_center_points_sorted[:fov_1_end]
            )

            # however, fov 1 sorted entries should NOT equal fov 1 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[:fov_1_end] != actual_center_points_sorted[:fov_1_end]

            # ensure the random center points for fov 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[fov_1_end:],
                actual_center_points=actual_center_points_sorted[fov_1_end:]
            )

            # however, fov 2 sorted entries should NOT equal fov 2 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[fov_1_end:] != actual_center_points_sorted[fov_1_end:]


def test_tma_generate_fov_list():
    # file path validation
    with pytest.raises(FileNotFoundError):
        tiling_utils.tma_generate_fov_list(
            'bad_path.json', 3, 3
        )

    # generate a sample FOVs list
    # NOTE: this intentionally contains more than 2 FOVs for now so it fails immediately
    # we will trim it later on
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100), (100, 100), (200, 200)],
        fov_names=["TheFirstFOV", "TheFirstFOV", "TheSecondFOV", "TheSecondFOV"]
    )

    # save sample FOV
    with open('sample_fovs_list.json', 'w') as sfl:
        json.dump(sample_fovs_list, sfl)

    # too few x-fovs defined
    with pytest.raises(ValueError):
        tiling_utils.tma_generate_fov_list(
           'sample_fovs_list.json', 2, 3
        )

    # too few y-fovs defined
    with pytest.raises(ValueError):
        tiling_utils.tma_generate_fov_list(
            'sample_fovs_list.json', 3, 2
        )

    # the fovs list defined does not contain exactly 2 FOVs
    with pytest.raises(ValueError):
        tiling_utils.tma_generate_fov_list(
            'sample_fovs_list.json', 3, 3
        )

    # trim sample_fovs_list so it only contains 2 FOVs
    # one to define the upper-left corner, one to define the bottom-left corner
    sample_fovs_list['fovs'] = sample_fovs_list['fovs'][:2]

    # error checking, define the upper-left x-coord to be greater than the bottom-right
    sample_fovs_list['fovs'][0]['centerPointMicrons']['x'] = 100
    sample_fovs_list['fovs'][1]['centerPointMicrons']['x'] = 0

    # resave the data
    with open('sample_fovs_list.json', 'w') as sfl:
        json.dump(sample_fovs_list, sfl)

    # assert test fails
    with pytest.raises(ValueError):
        tiling_utils.tma_generate_fov_list(
            'sample_fovs_list.json', 3, 4
        )

    # reset x values
    sample_fovs_list['fovs'][0]['centerPointMicrons']['x'] = 0
    sample_fovs_list['fovs'][1]['centerPointMicrons']['x'] = 100

    # error checking, define the upper-left y-coord to be less than the bottom-right
    sample_fovs_list['fovs'][0]['centerPointMicrons']['y'] = 0
    sample_fovs_list['fovs'][1]['centerPointMicrons']['y'] = 100

    # resave the data
    with open('sample_fovs_list.json', 'w') as sfl:
        json.dump(sample_fovs_list, sfl)

    # assert test fails
    with pytest.raises(ValueError):
        tiling_utils.tma_generate_fov_list(
            'sample_fovs_list.json', 3, 4
        )

    # reset y values
    sample_fovs_list['fovs'][0]['centerPointMicrons']['y'] = 100
    sample_fovs_list['fovs'][1]['centerPointMicrons']['y'] = 0

    # resave the data
    with open('sample_fovs_list.json', 'w') as sfl:
        json.dump(sample_fovs_list, sfl)

    # create the FOV regions
    # use 3 and 4 since 3 divides into clean ints and 4 does not
    fov_regions = tiling_utils.tma_generate_fov_list(
        'sample_fovs_list.json', 3, 4
    )

    center_points = [fov_regions[fov] for fov in fov_regions]

    # define the actual center points created
    # NOTE: y-coordinates are intentionally rounded
    actual_x = [0, 50, 100]
    actual_y = [100, 66, 33, 0]
    actual_center_points = [
        (x, y) for x in actual_x for y in actual_y
    ]

    # assert the centroids are the same
    assert center_points == actual_center_points


def test_convert_microns_to_pixels():
    # just need to test it gets the right values for one coordinate in microns
    sample_coord = (25000, 35000)
    new_coord = tiling_utils.convert_microns_to_pixels(sample_coord)

    assert new_coord == (612, 762)


def test_assign_closest_fovs():
    # define the coordinates and fov names generated from the fov script
    # note that we intentionally define more auto fovs than manual fovs
    # to test that not all auto fovs necessarily get mapped to
    auto_coords = [(0, 0), (0, 50), (0, 100), (100, 0), (100, 50), (100, 100),
                   (150, 100), (150, 150)]
    auto_fov_names = ['row%d_col%d' % (x, y) for (x, y) in auto_coords]

    # generate the list of automatically-generated fovs
    auto_sample_fovs = dict(zip(auto_fov_names, auto_coords))

    # define the coordinates and fov names proposed by the user
    manual_coords = [(0, 25), (50, 25), (50, 50), (75, 50), (100, 25)]
    manual_fov_names = ['row%d_col%d' % (x, y) for (x, y) in manual_coords]

    # generate the list of manual fovs
    manual_sample_fovs = test_utils.generate_sample_fovs_list(
        manual_coords, manual_fov_names
    )

    # generate the mapping from manual to automatically-generated
    manual_to_auto_map, manual_fovs_info, auto_fovs_info = \
        tiling_utils.assign_closest_fovs(
            manual_sample_fovs, auto_sample_fovs
        )

    # for each manual fov, ensure the centroids are the same in manual_fovs_info
    for fov in manual_sample_fovs['fovs']:
        manual_centroid = tiling_utils.convert_microns_to_pixels(
            tuple(fov['centerPointMicrons'].values())
        )

        assert manual_fovs_info[fov['name']] == manual_centroid

    # same for automatically-generated fovs
    for fov in auto_sample_fovs:
        auto_centroid = tiling_utils.convert_microns_to_pixels(
            auto_sample_fovs[fov]
        )

        assert auto_fovs_info[fov] == auto_centroid

    # assert the mapping is correct, this covers 2 other test cases:
    # 1. Not all auto fovs (ex. row150_col100 and row150_col150) will be mapped to
    # 2. Multiple manual fovs can map to one auto fov (ex. row0_col25 and row50_col25 to row0_col0)
    actual_map = {
        'row0_col25': 'row0_col0',
        'row50_col25': 'row0_col0',
        'row50_col50': 'row0_col100',
        'row75_col50': 'row100_col100',
        'row100_col25': 'row100_col0'
    }

    assert manual_to_auto_map == actual_map


def test_generate_fov_circles():
    # we'll literally be copying the data generated from test_assign_closest_fovs
    sample_manual_to_auto_map = {
        'row0_col25': 'row0_col0',
        'row50_col25': 'row0_col0',
        'row50_col50': 'row0_col50',
        'row75_col50': 'row100_col50',
        'row100_col25': 'row100_col0'
    }

    sample_manual_fovs_info = {
        'row0_col25': (0, 25),
        'row50_col25': (50, 25),
        'row50_col50': (50, 50),
        'row75_col50': (75, 50),
        'row100_col25': (100, 25)
    }

    sample_auto_fovs_info = {
        'row0_col0': (0, 0),
        'row0_col50': (0, 50),
        'row0_col100': (0, 100),
        'row100_col0': (100, 0),
        'row100_col50': (100, 50),
        'row100_col100': (100, 100)
    }

    # define the sample slide image
    sample_slide_img = np.full((200, 200, 3), 255)

    # draw the circles
    sample_slide_img = tiling_utils.generate_fov_circles(
        sample_manual_to_auto_map, sample_manual_fovs_info,
        sample_auto_fovs_info, 'row0_col25', 'row0_col0',
        sample_slide_img, draw_radius=1
    )

    # assert the centroids are correct and they are filled in
    for pti in sample_manual_fovs_info:
        x, y = sample_manual_fovs_info[pti]

        # dark red if row0_col25, else bright red
        if pti == 'row0_col25':
            assert np.all(sample_slide_img[x, y, :] == np.array([210, 37, 37]))
        else:
            assert np.all(sample_slide_img[x, y, :] == np.array([255, 133, 133]))

    # same for the auto annotations
    for ati in sample_auto_fovs_info:
        x, y = sample_auto_fovs_info[ati]

        # dark blue if row0_col0, else bright blue
        if ati == 'row0_col0':
            assert np.all(sample_slide_img[x, y, :] == np.array([50, 115, 229]))
        else:
            assert np.all(sample_slide_img[x, y, :] == np.array([162, 197, 255]))


@pytest.mark.parametrize('randomize_setting', _REMAP_RANDOMIZE_TEST_CASES)
@pytest.mark.parametrize('moly_insert', _REMAP_MOLY_INSERT_CASES)
@pytest.mark.parametrize('moly_interval', _REMAP_MOLY_INTERVAL_CASES)
def test_remap_and_reorder_fovs(randomize_setting, moly_insert, moly_interval):
    # error check: moly_path must exist
    with pytest.raises(FileNotFoundError):
        tiling_utils.remap_and_reorder_fovs({}, {}, 'bad_path.json')

    # define the sample Moly point
    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # save the Moly point
    with open('sample_moly_point.json', 'w') as smp:
        json.dump(sample_moly_point, smp)

    # error check: moly_interval must be at least 1
    with pytest.raises(ValueError):
        tiling_utils.remap_and_reorder_fovs({}, {}, 'sample_moly_point.json', moly_interval=0)

    # define the coordinates and fov names manual by the user
    manual_coords = [(0, 25), (50, 25), (50, 50), (75, 50), (100, 25), (100, 75)]
    manual_fov_names = ['row%d_col%d' % (x, y) for (x, y) in manual_coords]

    # generate the list of manual fovs
    manual_sample_fovs = test_utils.generate_sample_fovs_list(
        manual_coords, manual_fov_names
    )

    # define a sample mapping
    sample_mapping = {
        'row0_col25': 'row0_col0',
        'row50_col25': 'row0_col25',
        'row50_col50': 'row0_col100',
        'row75_col50': 'row100_col100',
        'row100_col25': 'row100_col0',
        'row100_col75': 'row100_col75'
    }

    # copy the data so it doesn't overwrite manual_sample_fovs
    manual_sample_fovs_copy = copy.deepcopy(manual_sample_fovs)

    # add id, name, and status
    manual_sample_fovs_copy['id'] = -1
    manual_sample_fovs_copy['name'] = 'test'
    manual_sample_fovs_copy['status'] = 'all_systems_go'

    # remap the FOVs
    remapped_sample_fovs = tiling_utils.remap_and_reorder_fovs(
        manual_sample_fovs_copy, sample_mapping, 'sample_moly_point.json', randomize_setting,
        moly_insert, moly_interval
    )

    # assert id, name, and status are the same
    assert remapped_sample_fovs['id'] == manual_sample_fovs_copy['id']
    assert remapped_sample_fovs['name'] == manual_sample_fovs_copy['name']
    assert remapped_sample_fovs['status'] == manual_sample_fovs_copy['status']

    # assert the same FOVs in the manual-to-auto map (sample_mapping)
    # appear in remapped sample FOVs after the remapping process
    misc_utils.verify_same_elements(
        remapped_fov_names=[fov['name'] for fov in remapped_sample_fovs['fovs']
                            if fov['name'] != 'MoQC'],
        fovs_in_mapping=list(sample_mapping.values())
    )

    # assert the mapping was done correctly
    scrambled_names = [fov['name'] for fov in remapped_sample_fovs['fovs']]
    for fov in manual_sample_fovs['fovs']:
        mapped_name = sample_mapping[fov['name']]
        assert mapped_name in scrambled_names

    # assert the same FOV coords are contained in remapped_sample_fovs as manual_sample_fovs
    scrambled_coords = [(fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
                        for fov in remapped_sample_fovs['fovs'] if fov['name'] != 'MoQC']
    misc_utils.verify_same_elements(
        scrambled_fov_coords=scrambled_coords,
        actual_coords=manual_coords
    )

    # enforce order–or not–depending on if randomization is added or not
    # NOTE: the randomization test fails once in a blue moon due to how randomization works
    if randomize_setting:
        assert scrambled_coords != manual_coords
    else:
        assert scrambled_coords == manual_coords

    # if Moly points will be inserted, assert they are in the right place at the right interval
    # otherwise, assert no Moly points appear
    if moly_insert:
        # assert the moly_indices are inserted at the correct locations
        moly_indices = np.arange(
            moly_interval, len(remapped_sample_fovs['fovs']), moly_interval + 1
        )
        for mi in moly_indices:
            assert remapped_sample_fovs['fovs'][mi]['name'] == 'MoQC'
    else:
        fov_names = [remapped_sample_fovs['fovs'][i]['name']
                     for i in range(len(remapped_sample_fovs['fovs']))]
        assert 'MoQC' not in fov_names
