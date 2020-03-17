from segmentation.utils import data_utils
import xarray as xr
import numpy as np
import os
import math
import pytest

import importlib
importlib.reload(data_utils)


def test_load_imgs_from_dir():

    # check default loading of all files
    test_path = "segmentation/tests/test_points_dir"
    test_loaded_xr = data_utils.load_imgs_from_dir(test_path, img_sub_folder="TIFs", dtype="int16")

    all_fovs = os.listdir(test_path)
    all_fovs = [fov for fov in all_fovs if "Point" in fov]

    all_imgs = os.listdir(os.path.join(test_path, "Point1", "TIFs"))
    all_imgs = [img for img in all_imgs if ".tif" in img]
    all_chans = [chan.split(".tif")[0] for chan in all_imgs]

    # make sure all folders loaded
    assert np.array_equal(test_loaded_xr.fovs, all_fovs)

    # make sure all channels loaded
    assert np.array_equal(test_loaded_xr.channels, all_chans)

    # check loading of specific files
    some_fovs = all_fovs[:2]
    some_imgs = all_imgs[:2]
    some_chans = [chan.split(".tif")[0] for chan in some_imgs]

    test_subset_xr = data_utils.load_imgs_from_dir(test_path, img_sub_folder="TIFs", dtype="int16",
                                                          folder_names=some_fovs, imgs=some_imgs)

    # make sure specified folders loaded
    assert np.array_equal(test_subset_xr.fovs, some_fovs)

    # make sure specified channels loaded
    assert np.array_equal(test_subset_xr.channels, some_chans)


    # make sure that load axis can be specified
    test_loaded_xr = data_utils.load_imgs_from_dir(test_path, img_sub_folder="TIFs", dtype="int16",
                                                   load_axis="stacks")
    assert(test_loaded_xr.dims[0] == "stacks")


def test_combine_xarrays():
    # test combining along points axis
    xr1 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 3)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30), ["chan1", "chan2", "chan3"]],
                       dims=["points", "rows", "cols", "channels"])

    xr2 = xr.DataArray(np.random.randint(10, size=(2, 30, 30, 3)),
                       coords=[["Point4", "Point5"], range(30), range(30), ["chan1", "chan2", "chan3"]],
                       dims=["points", "rows", "cols", "channels"])

    xr_combined = data_utils.combine_xarrays((xr1, xr2), axis=0)
    assert xr_combined.shape == (5, 30, 30, 3)

    # test combining along channels axis
    xr1 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 3)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30), ["chan1", "chan2", "chan3"]],
                       dims=["points", "rows", "cols", "channels"])

    xr2 = xr.DataArray(np.random.randint(10, size=(3, 30, 30, 2)),
                       coords=[["Point1", "Point2", "Point3"], range(30), range(30), ["chan3", "chan4"]],
                       dims=["points", "rows", "cols", "channels"])

    xr_combined = data_utils.combine_xarrays((xr1, xr2), axis=-1)
    assert xr_combined.shape == (3, 30, 30, 5)


def test_reorder_xarray_channels():

    # test switching without blank channels
    test_input = np.random.randint(5, size=(2, 128, 128, 3))

    # channel 0 is 3x bigger, channel 2 is 3x smaller
    test_input[:, :, :, 0] *= 3
    test_input[:, :, :, 2] //= 3

    test_xr = xr.DataArray(test_input,
                           coords=[["Point1", "Point2"], range(test_input.shape[1]), range(test_input.shape[2]),
                                   ["chan0", "chan1", "chan2"]],
                           dims=["points", "rows", "cols", "channels"])

    channel_order = ["chan2", "chan1", "chan0"]
    new_xr = data_utils.reorder_xarray_channels(channel_order, test_xr, non_blank_channels=test_xr.channels)

    # confirm that labels are in correct order, and that values were switched as well
    assert np.array_equal(channel_order, new_xr.channels)
    assert np.sum(new_xr.loc[:, :, :, "chan0"]) > np.sum(new_xr.loc[:, :, :, "chan2"])


    # test switching with blank channels
    channel_order = ["chan1", "chan2", "chan666"]
    new_xr = data_utils.reorder_xarray_channels(channel_order, test_xr, non_blank_channels=["chan1", "chan2"])

    # make sure order was switched, and that blank channel is mostly empty
    assert np.array_equal(channel_order, new_xr.channels)
    assert np.sum(new_xr.loc[:, :, :, "chan666"]) / (new_xr.shape[1] * new_xr.shape[2]) < 0.05


def test_pad_xr_dims():
    test_input = np.zeros((2, 10, 10, 3))
    test_xr = xr.DataArray(test_input,
                           coords=[["Point1", "Point2"], range(test_input.shape[1]), range(test_input.shape[2]), ["chan0", "chan1", "chan2"]],
                           dims=["points", "rows", "cols", "channels"])

    padded_dims = ["points", "rows", "rows2", "cols", "cols2", "channels"]

    padded_xr = data_utils.pad_xr_dims(test_xr, padded_dims)

    assert list(padded_xr.dims) == padded_dims

    # check that error raised when wrong dimensions
    padded_wrong_order_dims = ["rows", "points", "rows2", "cols", "cols2", "channels"]

    with pytest.raises(ValueError):
        data_utils.pad_xr_dims(test_xr, padded_wrong_order_dims)

def test_crop_helper():
    # test crops that divide evenly
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 128

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * (crop_input.shape[1] / crop_size) * (crop_input.shape[2] / crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test crops that don't divide evenly
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 100

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * math.ceil(crop_input.shape[1] / crop_size) * math.ceil(crop_input.shape[2] / crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))


def test_crop_image_stack():
    # test without overlap (stride_fraction = 1)
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 128
    stride_fraction = 1

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * math.floor(crop_input.shape[1] / crop_size) * math.floor(crop_input.shape[2] / crop_size) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test with overlap
    crop_input = np.zeros((4, 1024, 1024, 4))
    crop_size = 128
    stride_fraction = 0.25

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * math.floor(crop_input.shape[1] / crop_size) * math.floor(
        crop_input.shape[2] / crop_size) * (1 / stride_fraction) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))


