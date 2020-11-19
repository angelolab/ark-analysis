import os
import pytest
import tempfile

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
