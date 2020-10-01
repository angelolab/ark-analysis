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

    # test combining along channels axis
    fov_ids, chan_ids = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=5)

    base_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fov_ids, channel_names=chan_ids)

    test_xr = misc_utils.combine_xarrays((base_xr[:, :, :, :3], base_xr[:, :, :, 3:]), axis=-1)
    assert test_xr.equals(base_xr)


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
