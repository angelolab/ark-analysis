import xarray as xr
import numpy as np
import os
import math
import pytest
import tempfile
from shutil import rmtree

from mibidata import mibi_image as mi, tiff

from ark.utils import data_utils, test_utils
import skimage.io as io


def test_load_imgs_from_mibitiff():

    with tempfile.TemporaryDirectory() as temp_dir:

        # check unspecified point loading
        fovs = ["Point8_otherinfo"]
        channels = ["HH3", "Membrane"]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='mibitiff', delimiter='_',
            dtype=np.uint16
        )

        loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                       channels=channels,
                                                       delimiter='_')

        assert data_xr.equals(loaded_xr)

        test_utils.clear_directory(temp_dir)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "example_dataset",
                            "input_data", "mibitiff_inputs")
    channels = ["HH3", "Membrane"]

    # check specified point loading
    mibitiff_files = ["Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff"]
    data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                 mibitiff_files=mibitiff_files,
                                                 channels=channels,
                                                 delimiter='_')
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == channels).all()
    np.testing.assert_array_equal(
        data_xr.values[0],
        (tiff.read(os.path.join(data_dir, mibitiff_files[0])))[channels].data)

    with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir:
        tif = mi.MibiImage(np.random.rand(1024, 1024, 2).astype(np.float32),
                           ((1, channels[0]), (2, channels[1])),
                           **test_utils.METADATA)
        tiff.write(os.path.join(temp_dir, 'Point9.tiff'), tif, dtype=np.float32)
        tiff.write(os.path.join(temp_dir, 'Point8_junktext.tiff'), tif, dtype=np.float32)

        mibitiff_files = ['Point8_junktext.tiff', 'Point9.tiff']

        # test delimiter agnosticism
        data_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                     mibitiff_files=mibitiff_files,
                                                     channels=channels,
                                                     delimiter='_',
                                                     dtype=np.float32)

        assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
        assert(set(data_xr.fovs.values) == set(["Point8", "Point9"]))
        assert(data_xr.rows == range(1024)).all()
        assert(data_xr.cols == range(1024)).all()
        assert(data_xr.channels == channels).all()
        assert(np.issubdtype(data_xr.dtype, np.floating))

        # test float overwrite
        with pytest.warns(UserWarning):
            data_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                         mibitiff_files=[mibitiff_files[-1]],
                                                         channels=channels,
                                                         delimiter='_',
                                                         dtype='int16')

            assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
            assert(data_xr.fovs == "Point9")
            assert(data_xr.rows == range(1024)).all()
            assert(data_xr.cols == range(1024)).all()
            assert(data_xr.channels == channels).all()
            assert(np.issubdtype(data_xr.dtype, np.floating))


def test_load_imgs_from_mibitiff_all_channels():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "example_dataset",
                            "input_data", "mibitiff_inputs")
    mibitiff_files = ["Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff"]

    data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                 mibitiff_files=mibitiff_files,
                                                 channels=None,
                                                 delimiter='_')
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
    np.testing.assert_array_equal(
        data_xr.values[0],
        (tiff.read(os.path.join(data_dir, mibitiff_files[0]))).data)


def test_load_imgs_from_multitiff():
    # test all channels load w/ specified files
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "example_dataset",
                            "input_data", "deepcell_input")
    multitiff_files = ["Point8.tif"]
    data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                  multitiff_files=multitiff_files,
                                                  channels=None,
                                                  delimiter='_')
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == range(2)).all()

    # test single channel load
    data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                  multitiff_files=multitiff_files,
                                                  channels=[0],
                                                  delimiter='_')
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == [0]).all()

    # test all channels w/ unspecified files
    data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                  multitiff_files=None,
                                                  channels=None,
                                                  delimiter='_')
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == range(2)).all()

    with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir:
        tif = np.random.rand(1024, 1024, 2).astype('float')

        io.imsave(os.path.join(temp_dir, 'Point8.tif'),
                  tif,
                  plugin='tifffile')
        io.imsave(os.path.join(temp_dir, 'Point9_junktext.tif'),
                  tif,
                  plugin='tifffile')

        # test delimiter agnosticism
        data_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                      multitiff_files=None,
                                                      channels=None,
                                                      delimiter='_',
                                                      dtype='float')

        assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
        assert(set(data_xr.fovs.values) == set(["Point8", "Point9"]))
        assert(data_xr.rows == range(1024)).all()
        assert(data_xr.cols == range(1024)).all()
        assert(data_xr.channels == range(2)).all()
        assert(np.issubdtype(data_xr.dtype, np.floating))

        # test float overwrite
        with pytest.warns(UserWarning):
            data_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                          multitiff_files=['Point9_junktext.tif'],
                                                          channels=None,
                                                          delimiter='_',
                                                          dtype='int16')

            assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
            assert(data_xr.fovs == "Point9")
            assert(data_xr.rows == range(1024)).all()
            assert(data_xr.cols == range(1024)).all()
            assert(data_xr.channels == range(2)).all()
            assert(np.issubdtype(data_xr.dtype, np.floating))


def test_load_imgs_from_tree():
    # test loading from within fov directories
    with tempfile.TemporaryDirectory(prefix='fovs') as temp_dir:
        fovs = ["fov1", "fov2", "fov3"]
        imgs = ["img1.tiff", "img2.tiff", "img3.tiff"]
        chans = [chan.split(".tiff")[0] for chan in imgs]
        test_utils._create_img_dir(temp_dir, fovs, imgs)

        # check default loading of all files
        test_loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

        # make sure all folders loaded
        assert np.array_equal(test_loaded_xr.fovs.values.sort(), fovs.sort())

        # make sure all channels loaded
        assert np.array_equal(test_loaded_xr.channels.values.sort(), chans.sort())

        # check loading of specific files
        some_fovs = fovs[:2]
        some_imgs = imgs[:2]
        some_chans = chans[:2]

        test_subset_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           fovs=some_fovs, channels=some_imgs)

        # make sure specified folders loaded
        assert np.array_equal(test_subset_xr.fovs.values.sort(), some_fovs.sort())

        # make sure specified channels loaded
        assert np.array_equal(test_subset_xr.channels.values.sort(), some_chans.sort())

        # check loading w/o file extension
        test_noext_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=some_chans)

        # make sure all folders loaded
        assert np.array_equal(test_noext_xr.fovs.values.sort(), fovs.sort())

        # make sure specified channels loaded
        assert np.array_equal(test_noext_xr.channels.values.sort(), some_chans.sort())

        # check mixed extension presence
        test_someext_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=[chans[i] if i % 2 else imgs[i]
                                                     for i in range(3)])

        # make sure all folders loaded
        assert np.array_equal(test_someext_xr.fovs.values.sort(), fovs.sort())

        # makes sure all channels loaded
        assert np.array_equal(test_someext_xr.channels.values.sort(), chans.sort())

        # resave img3 as floats and test for float warning
        tif = np.random.rand(1024, 1024).astype("float")
        io.imsave(os.path.join(temp_dir, fovs[-1], "TIFs", imgs[-1]), tif)

        with pytest.warns(UserWarning):
            test_warning_xr = \
                data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                               fovs=[fovs[-1]], channels=[imgs[-1]])

            # test swap int16 -> float
            assert np.issubdtype(test_warning_xr.dtype, np.floating)


def test_load_imgs_from_dir():
    # test loading from 'free' directory
    with tempfile.TemporaryDirectory(prefix='one_file') as temp_dir:
        imgs = ["fov1_img1.tiff", "fov2_img2.tiff", "fov3_img3.tiff"]
        fovs = [img.split("_")[0] for img in imgs]
        test_utils._create_img_dir(temp_dir, fovs=[""], imgs=imgs, img_sub_folder="",
                                   dtype="float")

        # check default loading
        test_loaded_xr = \
            data_utils.load_imgs_from_dir(temp_dir, delimiter='_', dtype="float")

        # make sure grouping by file prefix was effective
        assert np.array_equal(test_loaded_xr.fovs, fovs)

        # make sure dim and coord were named w/ defaults
        assert np.all(test_loaded_xr.loc["fov1", :, :, "img_data"] >= 0)
        assert test_loaded_xr.dims[-1] == 'compartments'

        with pytest.warns(UserWarning):
            test_warning_xr = \
                data_utils.load_imgs_from_dir(temp_dir, delimiter='_', dtype="int16")

            # test swap int16 -> float
            assert np.issubdtype(test_warning_xr.dtype, np.floating)


def test_generate_deepcell_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['fov1', 'fov2']
        chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

        fov1path = os.path.join(temp_dir, 'fov1.tif')
        fov2path = os.path.join(temp_dir, 'fov2.tif')

        img_data = np.ones((2, 1024, 1024, 4), dtype="int16")
        img_data[0, :, :, 1] += 1
        img_data[0, :, :, 3] += 2

        data_xr = xr.DataArray(img_data, coords=[fovs, range(1024), range(1024), chans],
                               dims=["fovs", "rows", "cols", "channels"])

        # test 1 nuc, 1 mem (no summing)
        nucs = ['nuc2']
        mems = ['mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        # check shape
        assert fov1.shape == (1024, 1024, 2)
        assert fov2.shape == (1024, 1024, 2)

        assert np.all(fov1[:, :, 0] == 2)
        assert np.all(fov1[:, :, 1] == 3)
        assert np.all(fov2[:, :, 0] == 1)
        assert np.all(fov2[:, :, 1] == 1)

        # test 2 nuc, 2 mem (summing)
        nucs = ['nuc1', 'nuc2']
        mems = ['mem1', 'mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        assert np.all(fov1[:, :, 0] == 3)
        assert np.all(fov1[:, :, 1] == 4)
        assert np.all(fov2[:, :, 0] == 2)
        assert np.all(fov2[:, :, 1] == 2)

        # test nuc None
        nucs = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        # check shape (important for a None case)
        assert fov1.shape == (1024, 1024, 2)
        assert fov2.shape == (1024, 1024, 2)

        assert np.all(fov1[:, :, 0] == 0)
        assert np.all(fov1[:, :, 1] == 4)
        assert np.all(fov2[:, :, 0] == 0)
        assert np.all(fov2[:, :, 1] == 2)

        # test mem None
        nucs = ['nuc2']
        mems = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        assert np.all(fov1[:, :, 0] == 2)
        assert np.all(fov1[:, :, 1] == 0)
        assert np.all(fov2[:, :, 0] == 1)
        assert np.all(fov2[:, :, 1] == 0)


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


def test_combine_point_directories():
    # first test the case where the directory specified doesn't exist
    with pytest.raises(ValueError):
        data_utils.combine_point_directories(os.path.join("path", "to", "undefined", "folder"))

    # now we do the "real" testing...
    with tempfile.TemporaryDirectory() as temp_dir:
        os.mkdir(os.path.join(temp_dir, "test"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1", "point1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2", "point2"))

        data_utils.combine_point_directories(os.path.join(temp_dir, "test"))

        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir1_point1"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir2_point2"))


def test_stitch_images():
    fovs = ['fov' + str(i) for i in range(40)]
    chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

    img_data = np.ones((40, 10, 10, 4), dtype="int16")
    img_data[0, :, :, 1] += 1
    img_data[0, :, :, 3] += 2

    data_xr = xr.DataArray(img_data, coords=[fovs, range(10), range(10), chans],
                           dims=["fovs", "rows", "cols", "channels"])

    stitched_xr = data_utils.stitch_images(data_xr, 5)

    assert stitched_xr.shape == (1, 40 / 5 * 10, 40 / 8 * 10, 4)


def test_split_img_stack():
    with tempfile.TemporaryDirectory() as temp_dir:
        stack_dir = os.path.join(temp_dir, "stack_sample")
        output_dir = os.path.join(temp_dir, "output_sample")
        stack_list = ["channel_data.tif"]
        indices = [0, 1]
        names = ["chan1.tif", "chan2.tif"]

        os.mkdir(os.path.join(temp_dir, "stack_sample"))
        os.mkdir(os.path.join(temp_dir, "output_sample"))

        # first test channel_first=False
        junk_img_chan_last = np.zeros((1024, 1024, 10))
        io.imsave(os.path.join(stack_dir, "channel_data.tif"), junk_img_chan_last)

        data_utils.split_img_stack(stack_dir, output_dir, stack_list, indices, names,
                                   channels_first=False)

        assert os.path.exists(os.path.join(output_dir, "channel_data", "chan1.tif"))
        assert os.path.exists(os.path.join(output_dir, "channel_data", "chan2.tif"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "channel_data", "chan1.tif"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "channel_data", "chan2.tif"))

        assert sample_chan_1.shape == (1024, 1024)
        assert sample_chan_2.shape == (1024, 1024)

        # now overwrite old channel_data.jpg file and test channel_first=True
        junk_img_chan_first = np.zeros((10, 1024, 1024))
        io.imsave(os.path.join(stack_dir, "channel_data.tif"), junk_img_chan_first)

        # clear the original channel_data directory
        rmtree(os.path.join(output_dir, "channel_data"))

        data_utils.split_img_stack(stack_dir, output_dir, stack_list, indices, names,
                                   channels_first=True)

        assert os.path.exists(os.path.join(output_dir, "channel_data", "chan1.tif"))
        assert os.path.exists(os.path.join(output_dir, "channel_data", "chan2.tif"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "channel_data", "chan1.tif"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "channel_data", "chan2.tif"))

        assert sample_chan_1.shape == (1024, 1024)
        assert sample_chan_2.shape == (1024, 1024)
