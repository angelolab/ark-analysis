import os
import pytest
import tempfile

import numpy as np

from ark.utils import misc_utils, test_utils


def test_combine_xarrays():
    # test combining along fovs axis
    fov_ids, chan_ids = test_utils.gen_fov_chan_names(num_fovs=5, num_chans=3)

    base_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fov_ids, channel_names=chan_ids)

    test_xr = misc_utils.combine_xarrays((base_xr[:3, :, :, :], base_xr[3:, :, :, :]), axis=0)
    assert test_xr.equals(base_xr)

    # error checking
    with pytest.raises(ValueError):
        # pass xarrays with different fovs if axis is not 0
        misc_utils.combine_xarrays((base_xr[:3, :, :, :], base_xr[1:4, :, :, :]), axis=-1)

    # test combining along channels axis
    fov_ids, chan_ids = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=5)

    base_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fov_ids, channel_names=chan_ids)

    test_xr = misc_utils.combine_xarrays((base_xr[:, :, :, :3], base_xr[:, :, :, 3:]), axis=-1)
    assert test_xr.equals(base_xr)

    # error checking
    with pytest.raises(ValueError):
        # the two xarrays don't have the same dimensions
        misc_utils.combine_xarrays((base_xr[:, :, :, :3], base_xr[:, :, :, :2]), axis=0)

    with pytest.raises(ValueError):
        # pass xarrays with different channels if axis is 0
        misc_utils.combine_xarrays((base_xr[:, :, :, :3], base_xr[:, :, :, 1:4]), axis=0)


def test_combine_fov_directories():
    # first test the case where the directory specified doesn't exist
    with pytest.raises(ValueError):
        misc_utils.combine_fov_directories(os.path.join("path", "to", "undefined", "folder"))

    # now we do the "real" testing...
    with tempfile.TemporaryDirectory() as temp_dir:
        os.mkdir(os.path.join(temp_dir, "test"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1", "fov1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2", "fov2"))

        misc_utils.combine_fov_directories(os.path.join(temp_dir, "test"))

        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir1_fov1"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir2_fov2"))


def test_verify_in_list():
    with pytest.raises(ValueError):
        # not passing two lists to verify_in_list
        misc_utils.verify_in_list(one=['not_enough'])

    with pytest.raises(ValueError):
        # value is not contained in a list of acceptable values
        misc_utils.verify_in_list(one='hello', two=['goodbye', 'hello world'])

    with pytest.raises(ValueError):
        # not every element in a list is equal to an value
        misc_utils.verify_in_list(one=['goodbye', 'goodbye', 'hello'], two='goodbye')

    with pytest.raises(ValueError):
        # one list is not completely contained in another
        misc_utils.verify_in_list(one=['hello', 'world'],
                                  two=['hello', 'goodbye'])

    with pytest.warns(UserWarning):
        # issues a warning instead of raising an error
        misc_utils.verify_in_list(warn=True, one=['hello', 'world'],
                                  two=['hello', 'goodbye'])

    # test unwrapped string
    misc_utils.verify_in_list(
        one='hello',
        two=['hello', 'world']
    )

    # test numpy array allowance
    misc_utils.verify_in_list(
        one=np.array(['hello', 'world']),
        two=np.array(['hello', 'world', '!'])
    )


def test_verify_same_elements():
    with pytest.raises(ValueError):
        # not passing two lists to verify_same_elements
        misc_utils.verify_same_elements(one=['not_enough'])

    with pytest.raises(ValueError):
        # not passing in items that can be cast to list for either one or two
        misc_utils.verify_same_elements(one=1, two=2)

    with pytest.raises(ValueError):
        # the two lists provided do not contain the same elements
        misc_utils.verify_same_elements(one=['elem1', 'elem2', 'elem2'],
                                        two=['elem2', 'elem2', 'elem4'])

    with pytest.raises(ValueError):
        # the two lists provided differ in length (ordered checking)
        misc_utils.verify_same_elements(enforce_order=True,
                                        one=['elem1'],
                                        two=['elem1', 'elem2'])

    with pytest.raises(ValueError):
        # the two lists are ordered differently
        misc_utils.verify_same_elements(enforce_order=True,
                                        one=['elem1', 'elem2'],
                                        two=['elem2', 'elem1'])

    with pytest.warns(UserWarning):
        # issues a warning instead of raising an error
        misc_utils.verify_same_elements(warn=True, one=['elem1', 'elem2', 'elem2'],
                                        two=['elem2', 'elem2', 'elem4'])

    with pytest.warns(UserWarning):
        # issues a warning instead of raising an error for ordered
        misc_utils.verify_same_elements(enforce_order=True, warn=True,
                                        one=['elem1', 'elem2'],
                                        two=['elem2', 'elem1'])


def test_create_invalid_data_str():
    invalid_data = ["data_" + str(i) for i in range(20)]

    # Test to make sure the case of 10 invalid values creates a proper string.
    invalid_data_str1 = misc_utils.create_invalid_data_str(invalid_data=invalid_data[:10])
    for id in invalid_data[:10]:
        assert invalid_data_str1.find(id) != -1

    # Test to make sure cases of less than 10 invalid values creates a proper string.
    invalid_data_str2 = misc_utils.create_invalid_data_str(invalid_data=invalid_data[:3])
    for id in invalid_data[:3]:
        assert invalid_data_str2.find(id) != -1

    # Test to make sure cases of more than 10 invalid values creates a proper string
    # capping out at 10 values.
    invalid_data_str3 = misc_utils.create_invalid_data_str(invalid_data=invalid_data)
    for id in invalid_data[:10]:
        assert invalid_data_str3.find(id) != -1


def test_make_iterable():

    # Input is string, ignore_str = True
    input_str = "test"
    assert misc_utils.make_iterable(input_str, ignore_str=True) == ["test"]

    # Input is string, ignore_str = False
    assert misc_utils.make_iterable(input_str, ignore_str=False) == "test"

    # Input is a list of objects, ignore_str = True
    input_list = [1, 2, 3, 4]
    assert misc_utils.make_iterable(input_list, ignore_str=True) == [1, 2, 3, 4]

    # Input is a list of objects, ignore_str = False
    assert misc_utils.make_iterable(input_list, ignore_str=False) == [1, 2, 3, 4]

    # Input is of type(<...>) = type (a fundamental data type such as str, int, bool)
    assert misc_utils.make_iterable(str, ignore_str=True) == [str]

    assert misc_utils.make_iterable(str, ignore_str=False) == [str]

    assert misc_utils.make_iterable(int, ignore_str=True) == [int]

    assert misc_utils.make_iterable(int, ignore_str=False) == [int]

    assert misc_utils.make_iterable(bool, ignore_str=True) == [bool]

    assert misc_utils.make_iterable(bool, ignore_str=False) == [bool]

    # Input is a numpy array
    input_np_arr = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(misc_utils.make_iterable(input_np_arr, ignore_str=True),
                            np.array([1, 2, 3, 4, 5]))
    np.testing.assert_equal(misc_utils.make_iterable(input_np_arr, ignore_str=False),
                            np.array([1, 2, 3, 4, 5]))
