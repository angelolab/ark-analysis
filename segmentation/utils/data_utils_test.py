import xarray as xr
import numpy as np
import os
import pathlib
import math
import pytest
import tempfile
from mibidata import tiff

from segmentation.utils import data_utils
import skimage.io as io

import importlib

importlib.reload(data_utils)


def _create_img_dir(temp_dir, fovs, imgs, img_sub_folder="TIFs"):
    tif = np.random.randint(0, 100, 1024 ** 2).reshape((1024, 1024)).astype("int8")

    for fov in fovs:
        fov_path = os.path.join(temp_dir, fov, img_sub_folder)
        if not os.path.exists(fov_path):
            os.makedirs(fov_path)
        for img in imgs:
            io.imsave(os.path.join(fov_path, img), tif)


def test_validate_paths():

    with tempfile.TemporaryDirectory(dir='data') as valid_path:

        os.mkdir(valid_path + '/real_subdirectory')

        valid_parts = [p for p in pathlib.Path(valid_path).parts]
        valid_parts[0] = 'not_a_real_directory'

        starts_out_of_scope = os.path.join(*valid_parts)

        valid_parts[0] = 'data'
        valid_parts[1] = 'not_a_real_subdirectory'
        valid_parts.append('not_real_but_parent_is_problem')
        bad_middle_path = os.path.join(*valid_parts)

        valid_parts.pop()
        valid_parts[1] = 'real_subdirectory'
        wrong_file = os.path.join(valid_path, 'not_a_real_file.tiff')

        # test one valid path
        data_utils.validate_paths(valid_path)

        # test multiple valid paths
        data_utils.validate_paths([valid_path, 'data', valid_path+'/real_subdirectory'])

        # test out-of-scope
        with pytest.raises(ValueError, match=r".*not_a_real_directory*"):
            data_utils.validate_paths(starts_out_of_scope)

        # test mid-directory existence
        with pytest.raises(ValueError, match=r".*not_a_real_subdirectory*"):
            data_utils.validate_paths(bad_middle_path)

        # test file existence
        with pytest.raises(ValueError, match=r".*not_a_real_file*"):
            data_utils.validate_paths(wrong_file)


def test_load_imgs_from_mibitiff():
    mibitiff_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "data", "example_dataset",
                                   "input_data", "input_data_TIFF",
                                   "Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff")]
    channels = ["HH3", "Membrane"]
    data_xr = data_utils.load_imgs_from_mibitiff(mibitiff_files, channels)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == channels).all()
    np.testing.assert_array_equal(data_xr.values[0],
                                  (tiff.read(mibitiff_files[0]))[channels].data)


def test_load_imgs_from_mibitiff_all_channels():
    mibitiff_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "data", "example_dataset",
                                   "input_data", "input_data_TIFF",
                                   "Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff")]
    data_xr = data_utils.load_imgs_from_mibitiff(mibitiff_files, channels=None)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    exected_channels = ["Background", "BetaCatenin", "BetaTubulin", "CD20",
                        "CD3", "CD4", "CD45", "CD8", "CD9", "ECadherin", "ER",
                        "GLUT1", "HER2", "HH3", "HLA_Class_1", "Ki67",
                        "LaminAC", "Membrane", "NaK ATPase", "PanKeratin",
                        "SMA", "Vimentin"]
    assert(data_xr.channels == exected_channels).all()
    np.testing.assert_array_equal(data_xr.values[0],
                                  (tiff.read(mibitiff_files[0])).data)


def test_load_imgs_from_tree():
    # test loading from within fov directories
    with tempfile.TemporaryDirectory(prefix='fovs') as temp_dir:
        fovs = ["fov1", "fov2", "fov3"]
        imgs = ["img1.tiff", "img2.tiff", "img3.tiff"]
        chans = [chan.split(".tiff")[0] for chan in imgs]
        _create_img_dir(temp_dir, fovs, imgs)

        # check default loading of all files
        test_loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

        # make sure all folders loaded
        assert np.array_equal(test_loaded_xr.fovs, fovs)

        # make sure all channels loaded
        assert np.array_equal(test_loaded_xr.channels.values, chans)

        # check loading of specific files
        some_fovs = fovs[:2]
        some_imgs = imgs[:2]
        some_chans = chans[:2]

        test_subset_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           fovs=some_fovs, imgs=some_imgs)

        # make sure specified folders loaded
        assert np.array_equal(test_subset_xr.fovs, some_fovs)

        # make sure specified channels loaded
        assert np.array_equal(test_subset_xr.channels, some_chans)


def test_load_imgs_from_dir():
    # test loading from 'free' directory
    with tempfile.TemporaryDirectory(prefix='one_file') as temp_dir:
        imgs = ["fov1_img1.tiff", "fov2_img2.tiff", "fov3_img3.tiff"]
        fovs = [img.split("_")[0] for img in imgs]
        _create_img_dir(temp_dir, fovs=[""], imgs=imgs, img_sub_folder="")

        # check default loading
        test_loaded_xr = \
            data_utils.load_imgs_from_dir(temp_dir, delimiter='_')

        # make sure grouping by file prefix was effective
        assert np.array_equal(test_loaded_xr.fovs, fovs)


def test_combine_xarrays():
    # test combining along points axis
    xr1 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 3)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30),
                               ["chan1", "chan2", "chan3"]],
                       dims=["fovs", "rows", "cols", "channels"])

    xr2 = xr.DataArray(np.random.randint(10, size=(2, 30, 30, 3)),
                       coords=[["Point4", "Point5"], range(30), range(30),
                               ["chan1", "chan2", "chan3"]],
                       dims=["fovs", "rows", "cols", "channels"])

    xr_combined = data_utils.combine_xarrays((xr1, xr2), axis=0)
    assert xr_combined.shape == (5, 30, 30, 3)
    assert np.all(xr_combined.channels.values == xr1.channels.values)
    assert np.all(xr_combined.fovs == np.concatenate((xr1.fovs.values, xr2.fovs.values)))

    # test combining along channels axis
    xr1 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 3)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30),
                               ["chan1", "chan2", "chan3"]],
                       dims=["fovs", "rows", "cols", "channels"])

    xr2 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 2)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30),
                               ["chan3", "chan4"]],
                       dims=["fovs", "rows", "cols", "channels"])

    xr_combined = data_utils.combine_xarrays((xr1, xr2), axis=-1)
    assert xr_combined.shape == (3, 30, 30, 5)
    assert np.all(
        xr_combined.channels == np.concatenate((xr1.channels.values, xr2.channels.values)))
    assert np.all(xr_combined.fovs == xr1.fovs)


def test_crop_helper():
    # test crops that divide evenly
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 128

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * \
        (crop_input.shape[1] / crop_size) * (crop_input.shape[2] / crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test crops that don't divide evenly
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 100

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * math.ceil(crop_input.shape[1] / crop_size) * \
        math.ceil(crop_input.shape[2] / crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))


def test_crop_image_stack():
    # test without overlap (stride_fraction = 1)
    crop_input = np.zeros((4, 1024, 1024, 4), dtype="int16")
    crop_size = 128
    stride_fraction = 1

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * math.floor(crop_input.shape[1] / crop_size) * \
        math.floor(crop_input.shape[2] / crop_size) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test with overlap
    crop_input = np.zeros((4, 1024, 1024, 4), dtype="int16")
    crop_size = 128
    stride_fraction = 0.25

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * math.floor(crop_input.shape[1] / crop_size) * math.floor(
        crop_input.shape[2] / crop_size) * (1 / stride_fraction) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))
