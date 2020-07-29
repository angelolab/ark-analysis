import xarray as xr
import numpy as np
import os
import pathlib
import math
import pytest
import tempfile
from mibidata import mibi_image as mi, tiff


from segmentation.utils import data_utils
import skimage.io as io

import importlib

importlib.reload(data_utils)

# required metadata for mibitiff writing (barf)
METADATA = {
    'run': '20180703_1234_test', 'date': '2017-09-16T15:26:00',
    'coordinates': (12345, -67890), 'size': 500., 'slide': '857',
    'fov_id': 'Point1', 'fov_name': 'R1C3_Tonsil',
    'folder': 'Point1/RowNumber0/Depth_Profile0',
    'dwell': 4, 'scans': '0,5', 'aperture': 'B',
    'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
    'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
    'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
    'filename': '20180703_1234_test', 'description': 'test image',
    'version': 'alpha',
}


def _create_img_dir(temp_dir, fovs, imgs, img_sub_folder="TIFs", dtype="int8"):
    tif = np.random.randint(0, 100, 1024 ** 2).reshape((1024, 1024)).astype(dtype)

    for fov in fovs:
        fov_path = os.path.join(temp_dir, fov, img_sub_folder)
        if not os.path.exists(fov_path):
            os.makedirs(fov_path)
        for img in imgs:
            io.imsave(os.path.join(fov_path, img), tif)


def test_validate_paths():

    # change cwd to /scripts for more accurate testing
    os.chdir('scripts')

    # make a tempdir for testing
    with tempfile.TemporaryDirectory(dir='../data') as valid_path:

        # make valid subdirectory
        os.mkdir(valid_path + '/real_subdirectory')

        # extract parts of valid path to alter for test cases
        valid_parts = [p for p in pathlib.Path(valid_path).parts]
        valid_parts[0] = 'not_a_real_directory'

        # test no '../data' prefix
        starts_out_of_scope = os.path.join(*valid_parts)

        # construct test for bad middle folder path
        valid_parts[0] = '..'
        valid_parts[1] = 'data'
        valid_parts[2] = 'not_a_real_subdirectory'
        valid_parts.append('not_real_but_parent_is_problem')
        bad_middle_path = os.path.join(*valid_parts)

        # construct test for real path until file
        wrong_file = os.path.join(valid_path + '/real_subdirectory', 'not_a_real_file.tiff')

        # test one valid path
        data_utils.validate_paths(valid_path)

        # test multiple valid paths
        data_utils.validate_paths([valid_path, '../data', valid_path + '/real_subdirectory'])

        # test out-of-scope
        with pytest.raises(ValueError, match=r".*not_a_real_directory.*prefixed.*"):
            data_utils.validate_paths(starts_out_of_scope)

        # test mid-directory existence
        with pytest.raises(ValueError, match=r".*bad path.*not_a_real_subdirectory.*"):
            data_utils.validate_paths(bad_middle_path)

        # test file existence
        with pytest.raises(ValueError, match=r".*The file/path.*not_a_real_file.*"):
            data_utils.validate_paths(wrong_file)

    # reset cwd after testing
    os.chdir('../')


# TODO: test '_' delimiter autodecode + dtype overload+warning
def test_load_imgs_from_mibitiff():
    # check unspecified point loading
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "example_dataset",
                            "input_data", "mibitiff_inputs")
    channels = ["HH3", "Membrane"]
    data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                 channels=channels)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == channels).all()

    # check specified point loading
    mibitiff_files = ["Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff"]
    data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                 mibitiff_files=mibitiff_files,
                                                 channels=channels)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == channels).all()
    np.testing.assert_array_equal(
        data_xr.values[0],
        (tiff.read(os.path.join(data_dir, mibitiff_files[0])))[channels].data)

    # this is looped to make sure temp_file doesn't have '_' in its name
    # (sadly there's no excluded characters option for tempfile)
    while True:
        with tempfile.NamedTemporaryFile(suffix=".tif", dir=data_dir) as temp_file:
            if '_' in temp_file.name.split(os.sep)[-1]:
                continue
            tif = mi.MibiImage(np.random.rand(1024, 1024, 2).astype(np.float32),
                               ((1, channels[0]), (2, channels[1])),
                               **METADATA)
            tiff.write(temp_file, tif, dtype=np.float32)
            tempname = temp_file.name.split(os.sep)[-1]

            mibitiff_files.append(tempname)

            # test delimiter agnosticism
            data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                         mibitiff_files=None,
                                                         channels=channels,
                                                         dtype=np.float32)

            assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
            assert(set(["Point8", tempname.split('.')[0]]) == set(data_xr.fovs.values))
            assert(data_xr.rows == range(1024)).all()
            assert(data_xr.cols == range(1024)).all()
            assert(data_xr.channels == channels).all()
            assert(np.issubdtype(data_xr.dtype, np.floating))

            # test float overwrite
            with pytest.warns(UserWarning):
                data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                             mibitiff_files=[tempname],
                                                             channels=channels,
                                                             dtype='int16')

                assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
                assert(data_xr.fovs == tempname.split('.')[0])
                assert(data_xr.rows == range(1024)).all()
                assert(data_xr.cols == range(1024)).all()
                assert(data_xr.channels == channels).all()
                assert(np.issubdtype(data_xr.dtype, np.floating))

        break


def test_load_imgs_from_mibitiff_all_channels():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "example_dataset",
                            "input_data", "mibitiff_inputs")
    mibitiff_files = ["Point8_RowNumber0_Depth_Profile0-MassCorrected-Filtered.tiff"]

    data_xr = data_utils.load_imgs_from_mibitiff(data_dir,
                                                 mibitiff_files=mibitiff_files,
                                                 channels=None)
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
                                                  channels=None)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == range(2)).all()

    # test single channel load
    data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                  multitiff_files=multitiff_files,
                                                  channels=[0])
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == [0]).all()

    # test all channels w/ unspecified files
    data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                  multitiff_files=None,
                                                  channels=None)
    assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
    assert(data_xr.fovs == "Point8")
    assert(data_xr.rows == range(1024)).all()
    assert(data_xr.cols == range(1024)).all()
    assert(data_xr.channels == range(2)).all()

    with tempfile.NamedTemporaryFile(prefix="Point9_", suffix=".tif", dir=data_dir) as temp_file:
        tif = np.random.rand(1024, 1024, 2).astype('float')
        io.imsave(temp_file, tif, plugin='tifffile')
        tempname = temp_file.name.split(os.sep)[-1]
        multitiff_files = ["Point8.tif", tempname]

        # test delimiter agnosticism
        data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                      multitiff_files=multitiff_files,
                                                      channels=None,
                                                      dtype='float')

        assert(data_xr.dims == ("fovs", "rows", "cols", "channels"))
        assert(set(data_xr.fovs.values) == set(["Point8", "Point9"]))
        assert(data_xr.rows == range(1024)).all()
        assert(data_xr.cols == range(1024)).all()
        assert(data_xr.channels == range(2)).all()
        assert(np.issubdtype(data_xr.dtype, np.floating))

        # test float overwrite
        with pytest.warns(UserWarning):
            data_xr = data_utils.load_imgs_from_multitiff(data_dir,
                                                          multitiff_files=[tempname],
                                                          channels=None,
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
        assert np.array_equal(test_subset_xr.channels.values, some_chans)

        # check loading w/o file extension
        test_noext_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           imgs=some_chans)

        # make sure all folders loaded
        assert np.array_equal(test_noext_xr.fovs, fovs)

        # make sure specified channels loaded
        assert np.array_equal(test_noext_xr.channels.values, some_chans)

        # check mixed extension presence
        test_someext_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           imgs=[chans[i] if i % 2 else imgs[i] for i in range(3)])

        # make sure all folders loaded
        assert np.array_equal(test_someext_xr.fovs, fovs)

        # makes sure all channels loaded
        assert np.array_equal(test_someext_xr.channels.values, chans)

        # resave img3 as floats and test for float warning
        tif = np.random.rand(1024, 1024).astype("float")
        io.imsave(os.path.join(temp_dir, fovs[-1], "TIFs", imgs[-1]), tif)

        with pytest.warns(UserWarning):
            test_warning_xr = \
                data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                               fovs=[fovs[-1]], imgs=[imgs[-1]])

            # test swap int16 -> float
            assert np.issubdtype(test_warning_xr.dtype, np.floating)


def test_load_imgs_from_dir():
    # test loading from 'free' directory
    with tempfile.TemporaryDirectory(prefix='one_file') as temp_dir:
        imgs = ["fov1_img1.tiff", "fov2_img2.tiff", "fov3_img3.tiff"]
        fovs = [img.split("_")[0] for img in imgs]
        _create_img_dir(temp_dir, fovs=[""], imgs=imgs, img_sub_folder="", dtype="float")

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
