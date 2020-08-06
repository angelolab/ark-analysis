import numpy as np
import xarray as xr
import os
import pytest
import tempfile

import skimage.morphology as morph
import skimage.io as io
from skimage.morphology import erosion
from skimage.measure import regionprops

from mibidata import mibi_image as mi, tiff

from segmentation.utils import segmentation_utils
from segmentation.utils import data_utils_test


def _generate_deepcell_output(fov_num=2):
    fovs = ["fov" + str(i) for i in range(fov_num)]
    models = ["pixelwise_interior", "watershed_inner", "watershed_outer",
              "fgbg_foreground", "pixelwise_sum"]
    output = np.random.rand(len(fovs) * 50 * 50 * len(models))
    output = output.reshape((len(fovs), 50, 50, len(models)))

    output_xr = xr.DataArray(output, coords=[fovs, range(50), range(50), models],
                             dims=["fovs", "rows", "cols", "models"])
    return output_xr


def _generate_channel_xr(fov_num=2, chan_num=5):
    fovs = ["fov" + str(i) for i in range(fov_num)]
    channels = ["channel" + str(i) for i in range(chan_num)]
    output = np.random.randint(0, 20, len(fovs) * 50 * 50 * len(channels))
    output = output.reshape((len(fovs), 50, 50, len(channels)))

    output_xr = xr.DataArray(output, coords=[fovs, range(50), range(50), channels],
                             dims=["fovs", "rows", "cols", "channels"])

    return output_xr


def _create_test_extraction_data():
    # first create segmentation masks
    cell_mask = np.zeros((40, 40), dtype='int16')
    cell_mask[4:10, 4:10] = 1
    cell_mask[15:25, 20:30] = 2
    cell_mask[27:32, 3:28] = 3
    cell_mask[35:40, 15:22] = 4

    # then create channels data
    channel_data = np.zeros((40, 40, 5), dtype="int16")
    channel_data[:, :, 0] = 1
    channel_data[:, :, 1] = 5
    channel_data[:, :, 2] = 5
    channel_data[:, :, 3] = 10
    channel_data[:, :, 4] = 0

    # cell1 is the only cell negative for channel 3
    cell1 = cell_mask == 1
    channel_data[cell1, 3] = 0

    # cell2 is the only cell positive for channel 4
    cell2 = cell_mask == 2
    channel_data[cell2, 4] = 10

    return cell_mask, channel_data


def test_compute_complete_expression_matrices():
    # checks that a ValueError is thrown when the user tries to specify points that are not
    # in the original segmentation mask
    with pytest.raises(ValueError):
        # generate a segmentation array with 1 FOV
        cell_masks = np.zeros((1, 50, 50, 1), dtype="int16")

        segmentation_masks = xr.DataArray(cell_masks,
                                          coords=[["Point1"], range(50), range(50),
                                                  ["whole_cell"]],
                                          dims=["fovs", "rows", "cols", "compartments"])

        segmentation_utils.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, base_dir="path/to/base/dir", tiff_dir="path/to/tiff/dir",
            img_sub_folder="path/to/img/sub/folder", is_mibitiff=False, points=["Point1", "Point2"], batch_size=5)

    # checks if the tree loading is being called correctly when is_mibitiff is False
    # save the actual expression matrix and data loding tests for their respective test functions
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 2 FOVs and 2 imgs per FOV
        fovs = ["Point1", "Point2"]
        imgs = ["img1.tiff, img2.tiff"]

        # since example_dataset exists, lets create a new directory called testing_dataset
        # the rest of the directory structure will be the same
        base_dir = os.path.join(temp_dir, "testing_dataset")
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = os.path.join(input_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        # create the directory structure, with a little help from _create_img_dir
        os.mkdir(base_dir)
        os.mkdir(input_dir)
        os.mkdir(tiff_dir)
        data_utils_test._create_img_dir(temp_dir=tiff_dir, fovs=fovs, imgs=imgs, img_sub_folder=img_sub_folder, dtype="int16")

        # generate a sample segmentation_mask
        cell_mask, _ = _create_test_extraction_data()
        cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask
        cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]
        segmentation_masks = xr.DataArray(cell_masks,
                                          coords=[fovs, range(40), range(40),
                                                  ["whole_cell"]],
                                          dims=["fovs", "rows", "cols", "compartments"])

        # generate sample norm and arcsinh data
        norm_data, arcsinh_data = segmentation_utils.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, base_dir=base_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, points=fovs, batch_size=5)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0

    # checks if the loading is being called correctly when is_mibitiff is True
    # save the actual expression matrix and data loding tests for their respective test functions
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 2 FOVs and 2 mibitiff_imgs
        fovs = ["Point1", "Point2"]
        mibitiff_imgs = ["Point1_example_mibitiff.tiff", "Point2_example_mibitiff.tiff"]

        # since example_dataset exists, lets create a new directory called testing_dataset
        # the rest of the directory structure will be the same
        base_dir = os.path.join(temp_dir, "testing_dataset")
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = os.path.join(input_dir, "mibitiff_inputs")

        # create the directory structure
        os.mkdir(base_dir)
        os.mkdir(input_dir)
        os.mkdir(tiff_dir)

        # create sample mibitiff images for each point
        for f, m in zip(fovs, mibitiff_imgs):
            # required metadata for mibitiff writing (double barf)
            METADATA = {
                'run': '20180703_1234_test', 'date': '2017-09-16T15:26:00',
                'coordinates': (12345, -67890), 'size': 500., 'slide': '857',
                'fov_id': f, 'fov_name': 'R1C3_Tonsil',
                'folder': f + '/RowNumber0/Depth_Profile0',
                'dwell': 4, 'scans': '0,5', 'aperture': 'B',
                'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
                'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
                'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
                'filename': '20180703_1234_test', 'description': 'test image',
                'version': 'alpha',
            }

            channels = ["HH3", "Membrane"]
            sample_tif = mi.MibiImage(np.random.rand(1024, 1024, 2).astype(np.float32),
                                      ((1, channels[0]), (2, channels[1])),
                                      **METADATA)
            tiff.write(os.path.join(tiff_dir, m), sample_tif, dtype=np.float32)

        # generate a sample segmentation_mask
        cell_mask, _ = _create_test_extraction_data()
        cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask
        cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]
        segmentation_masks = xr.DataArray(cell_masks,
                                          coords=[fovs, range(40), range(40),
                                                  ["whole_cell"]],
                                          dims=["fovs", "rows", "cols", "compartments"])

        # generate sample norm and arcsinh data
        norm_data, arcsinh_data = segmentation_utils.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, base_dir=base_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=True, points=fovs, batch_size=5)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0


def test_watershed_transform():
    model_output = _generate_deepcell_output()
    channel_data = _generate_channel_xr()

    overlay_channels = [channel_data.channels.values[:2]]

    with tempfile.TemporaryDirectory() as temp_dir:
        # test default settings
        segmentation_utils.watershed_transform(model_output=model_output, channel_xr=channel_data,
                                               output_dir=temp_dir,
                                               overlay_channels=overlay_channels,
                                               interior_threshold=0.5)

        saved_output = xr.load_dataarray(os.path.join(temp_dir, 'segmentation_labels.xr'))

        # ensure sequential labeling
        for fov in range(saved_output.shape[0]):
            cell_num = len(np.unique(saved_output[fov, :, :, 0]))
            max_val = np.max(saved_output[fov, :, :, 0].values)
            assert cell_num == max_val + 1  # background counts towards unique cells

    with tempfile.TemporaryDirectory() as temp_dir:
        # test different networks settings
        segmentation_utils.watershed_transform(model_output=model_output, channel_xr=channel_data,
                                               output_dir=temp_dir,
                                               overlay_channels=overlay_channels,
                                               maxima_model="watershed_inner",
                                               interior_model="watershed_outer")

    with tempfile.TemporaryDirectory() as temp_dir:
        # test save_all tifs
        segmentation_utils.watershed_transform(model_output=model_output, channel_xr=channel_data,
                                               output_dir=temp_dir,
                                               overlay_channels=overlay_channels,
                                               save_tifs='all')
        assert os.path.exists(os.path.join(temp_dir, 'fov1_interior_smoothed.tiff'))

    with tempfile.TemporaryDirectory() as temp_dir:
        # only a subset
        segmentation_utils.watershed_transform(model_output=model_output, channel_xr=channel_data,
                                               output_dir=temp_dir,
                                               overlay_channels=overlay_channels,
                                               save_tifs='overlays')
        assert not os.path.exists(os.path.join(temp_dir, 'fov1_interior_smoothed.tiff'))

    with tempfile.TemporaryDirectory() as temp_dir:
        # test multiple different overlay_channels
        overlay_channels = [channel_data.channels.values[3:4], channel_data.channels.values[1:4]]
        segmentation_utils.watershed_transform(model_output=model_output, channel_xr=channel_data,
                                               output_dir=temp_dir,
                                               overlay_channels=overlay_channels)

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad arguments

        with pytest.raises(ValueError):
            # bad fov names
            fovs = [model_output.fovs.values[0], 'bad_name']
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels,
                                                   fovs=fovs)

        with pytest.raises(ValueError):
            # incomplete channel data
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data[1:, ...],
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels)

        with pytest.raises(ValueError):
            # bad overlay channels
            bad_overlay_channels = [[channel_data.channels.values[0], 'bad_channel']]
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=bad_overlay_channels)

        with pytest.raises(ValueError):
            # bad directory name
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data,
                                                   output_dir='bad_output_dir',
                                                   overlay_channels=overlay_channels)

        with pytest.raises(ValueError):
            # bad maxima model name
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels,
                                                   maxima_model='bad_maxima_model')

        with pytest.raises(ValueError):
            # maxima model not included
            missing_pixel_model = model_output[..., 2:]
            segmentation_utils.watershed_transform(model_output=missing_pixel_model,
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels)

        with pytest.raises(ValueError):
            # bad interior model name
            segmentation_utils.watershed_transform(model_output=model_output,
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels,
                                                   interior_model='bad_interior_model')

        with pytest.raises(ValueError):
            # maxima model not included
            missing_interior_model = model_output[..., 1:]
            segmentation_utils.watershed_transform(model_output=missing_interior_model,
                                                   maxima_model='watershed_inner',
                                                   channel_xr=channel_data,
                                                   output_dir=temp_dir,
                                                   overlay_channels=overlay_channels)


def test_find_nuclear_mask_id():
    # create cell labels with 5 distinct cells
    cell_labels = np.zeros((60, 10), dtype='int')
    for i in range(6):
        cell_labels[(i * 10):(i * 10 + 8), :8] = i + 1

    # create nuc labels with varying degrees of overlap
    nuc_labels = np.zeros((60, 60), dtype='int')

    # perfect overlap
    nuc_labels[:8, :8] = 1

    # greater than majority overlap
    nuc_labels[10:16, :6] = 2

    # only partial overlap
    nuc_labels[20:23, :3] = 3

    # no overlap for cell 4

    # two cells overlapping, larger cell_id correct
    nuc_labels[40:48, :2] = 5
    nuc_labels[40:48, 2:8] = 20

    # two cells overlapping, background is highest
    nuc_labels[50:58, :1] = 21
    nuc_labels[50:58, 1:3] = 6

    true_nuc_ids = [1, 2, 3, None, 20, 6]

    cell_props = regionprops(cell_labels)

    # check that predicted nuclear id is correct for all cells in image
    for idx, prop in enumerate(cell_props):
        predicted_nuc = segmentation_utils.find_nuclear_mask_id(nuc_segmentation_mask=nuc_labels,
                                                                cell_coords=prop.coords)

        assert predicted_nuc == true_nuc_ids[idx]


# TODO: refactor to avoid code reuse
def test_transform_expression_matrix():
    # create expression matrix
    cell_data = np.random.choice([0, 1, 2, 3, 4], 70, replace=True)
    cell_data = cell_data.reshape((1, 10, 7)).astype('float')

    coords = [['whole_cell'], list(range(10)),
              ['cell_size', 'chan1', 'chan2', 'chan3', 'label', 'morph_1', 'morph_2']]
    dims = ['compartments', 'cell_id', 'features']

    cell_data = xr.DataArray(cell_data, coords=coords, dims=dims)

    unchanged_cols = ['cell_size', 'label', 'morph_1', 'morph_2']
    modified_cols = ['chan1', 'chan2', 'chan3']

    # test size_norm
    normalized_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                     transform='size_norm')

    assert np.array_equal(normalized_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        if cell_data.loc['whole_cell', cell, 'cell_size'] != 0:
            normalized_vals = np.divide(cell_data.loc['whole_cell', cell, modified_cols].values,
                                        cell_data.loc['whole_cell', cell, 'cell_size'].values)
            assert np.array_equal(normalized_data.loc['whole_cell', cell, modified_cols].values,
                                  normalized_vals)

    # test arcsinh transform
    transform_kwargs = {'linear_factor': 1}
    arcsinh_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                  transform='arcsinh',
                                                                  transform_kwargs=transform_kwargs)

    assert np.array_equal(arcsinh_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        arcsinh_vals = np.arcsinh(cell_data.loc[:, cell, modified_cols].values)
        assert np.array_equal(arcsinh_data.loc[:, cell, modified_cols].values, arcsinh_vals)


def test_transform_expression_matrix_multiple_compartments():
    # create expression matrix
    cell_data = np.random.choice([0, 1, 2, 3, 4], 140, replace=True)
    cell_data = cell_data.reshape((2, 10, 7)).astype('float')

    coords = [['whole_cell', 'nuclear'], list(range(10)),
              ['cell_size', 'chan1', 'chan2', 'chan3', 'label', 'morph_1', 'morph_2']]
    dims = ['compartments', 'cell_id', 'features']

    cell_data = xr.DataArray(cell_data, coords=coords, dims=dims)

    unchanged_cols = ['cell_size', 'label', 'morph_1', 'morph_2']
    modified_cols = ['chan1', 'chan2', 'chan3']

    # test size_norm
    normalized_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                     transform='size_norm')

    assert np.array_equal(normalized_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        if cell_data.loc['whole_cell', cell, 'cell_size'] != 0:
            normalized_vals = np.divide(cell_data.loc['whole_cell', cell, modified_cols].values,
                                        cell_data.loc['whole_cell', cell, 'cell_size'].values)
            assert np.array_equal(normalized_data.loc['whole_cell', cell, modified_cols].values,
                                  normalized_vals)

    # test arcsinh transform
    transform_kwargs = {'linear_factor': 1}
    arcsinh_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                  transform='arcsinh',
                                                                  transform_kwargs=transform_kwargs)

    assert np.array_equal(arcsinh_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        arcsinh_vals = np.arcsinh(cell_data.loc[:, cell, modified_cols].values)
        assert np.array_equal(arcsinh_data.loc[:, cell, modified_cols].values, arcsinh_vals)


# TODO: The testing for this function can be improved. Repeated code, and lots of manual checks
def test_compute_marker_counts():
    cell_mask, channel_data = _create_test_extraction_data()

    # create xarray containing segmentation mask
    coords = [range(40), range(40), ['whole_cell']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks = xr.DataArray(np.expand_dims(cell_mask, axis=-1), coords=coords, dims=dims)

    # create xarray with channel data
    coords = [range(40), range(40), ['chan0', 'chan1', 'chan2', 'chan3', 'chan4']]
    dims = ['rows', 'cols', 'channels']
    input_images = xr.DataArray(channel_data, coords=coords, dims=dims)

    segmentation_output = \
        segmentation_utils.compute_marker_counts(input_images=input_images,
                                                 segmentation_masks=segmentation_masks)

    # check that channel 0 counts are same as cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'cell_size'].values,
                          segmentation_output.loc['whole_cell', :, 'chan0'].values)

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'cell_size'].values * 5,
                          segmentation_output.loc['whole_cell', :, 'chan1'].values)

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'chan2'].values,
                          segmentation_output.loc['whole_cell', :, 'chan1'].values)

    # check that only cell1 is negative for channel 3
    assert segmentation_output.loc['whole_cell', :, 'chan3'][1] == 0
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan3'][2:] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output.loc['whole_cell', :, 'chan4'][2] > 0
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan4'][:2] == 0)
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan4'][3:] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(cell_mask == cell_id) for cell_id in [1, 2, 3, 4]]
    assert np.array_equal(sizes, segmentation_output.loc['whole_cell', 1:, 'cell_size'])

    # check that regionprops size matches with cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', 1:, 'cell_size'],
                          segmentation_output.loc['whole_cell', 1:, 'area'])

    # test whole_cell and nuclear compartments with same data
    equal_masks = np.stack((cell_mask, cell_mask), axis=-1)
    # create xarray containing segmentation mask
    coords = [range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks_equal = xr.DataArray(equal_masks, coords=coords, dims=dims)

    segmentation_output_equal = \
        segmentation_utils.compute_marker_counts(input_images=input_images,
                                                 segmentation_masks=segmentation_masks_equal,
                                                 nuclear_counts=True)

    assert np.all(segmentation_output_equal[0].values == segmentation_output_equal[1].values)

    # test with different sized nuclear and whole_cell masks

    # nuclear mask is smaller
    nuc_mask = erosion(cell_mask, selem=morph.disk(1))

    unequal_masks = np.stack((cell_mask, nuc_mask), axis=-1)
    coords = [range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks_unequal = xr.DataArray(unequal_masks, coords=coords, dims=dims)

    segmentation_output_unequal = \
        segmentation_utils.compute_marker_counts(input_images=input_images,
                                                 segmentation_masks=segmentation_masks_unequal,
                                                 nuclear_counts=True)

    # make sure nuclear segmentations are smaller
    assert np.all(segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'].values <
                  segmentation_output_unequal.loc['whole_cell', 1:, 'cell_size'].values)

    # check that channel 0 counts are same as cell size
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values,
                          segmentation_output_unequal.loc['nuclear', :, 'chan0'].values)

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values * 5,
                          segmentation_output_unequal.loc['nuclear', :, 'chan1'].values)

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'chan2'].values,
                          segmentation_output_unequal.loc['nuclear', :, 'chan1'].values)

    # check that only cell1 is negative for channel 3
    assert segmentation_output_unequal.loc['nuclear', :, 'chan3'][1] == 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan3'][2:] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output_unequal.loc['nuclear', :, 'chan4'][2] > 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan4'][:2] == 0)
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan4'][3:] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(nuc_mask == cell_id) for cell_id in [1, 2, 3, 4]]
    assert np.array_equal(sizes, segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'])

    assert np.array_equal(segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'],
                          segmentation_output_unequal.loc['nuclear', 1:, 'area'])


# TODO: refactor these tests to share code
def test_generate_expression_matrix():
    cell_mask, channel_data = _create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask
    cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data
    channel_datas[1, 5:, 5:, :] = channel_data[:-5, :-5]

    segmentation_masks = xr.DataArray(cell_masks,
                                      coords=[["Point1", "Point2"], range(40), range(40),
                                              ["whole_cell"]],
                                      dims=["fovs", "rows", "cols", "compartments"])

    channel_data = xr.DataArray(channel_datas,
                                coords=[["Point1", "Point2"], range(40), range(40),
                                        ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                dims=["fovs", "rows", "cols", "channels"])

    normalized, arcsinh = segmentation_utils.generate_expression_matrix(segmentation_masks,
                                                                        channel_data)

    assert normalized.shape[0] == 7

    assert np.all(normalized['chan0'] == np.repeat(1, len(normalized)))
    assert np.all(normalized['chan1'] == np.repeat(5, len(normalized)))
    assert np.all(normalized['chan2'] == normalized['chan2'])


def test_generate_expression_matrix_multiple_compartments():
    cell_mask, channel_data = _create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask
    cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data
    channel_datas[1, 5:, 5:, :] = channel_data[:-5, :-5]

    # generate a second set of nuclear masks that are smaller than cell masks
    nuc_masks = np.zeros_like(cell_masks)
    nuc_masks[0, :, :, 0] = erosion(cell_masks[0, :, :, 0], selem=morph.disk(1))
    nuc_masks[1, :, :, 0] = erosion(cell_masks[1, :, :, 0], selem=morph.disk(1))

    # cell 2 in fov0 has no nucleus
    nuc_masks[0, nuc_masks[0, :, :, 0] == 2, 0] = 0

    unequal_masks = np.concatenate((cell_masks, nuc_masks), axis=-1)
    coords = [["Point0", "Point1"], range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['fovs', 'rows', 'cols', 'compartments']
    segmentation_masks_unequal = xr.DataArray(unequal_masks, coords=coords, dims=dims)

    channel_data = xr.DataArray(channel_datas,
                                coords=[["Point0", "Point1"], range(40), range(40),
                                        ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                dims=["fovs", "rows", "cols", "channels"])

    normalized, arcsinh = segmentation_utils.generate_expression_matrix(segmentation_masks_unequal,
                                                                        channel_data,
                                                                        nuclear_counts=True)

    assert normalized.shape[0] == 7

    assert np.all(normalized['chan0'] == np.repeat(1, len(normalized)))
    assert np.all(normalized['chan1'] == np.repeat(5, len(normalized)))
    assert np.all(normalized['chan2'] == normalized['chan2'])

    # check that missing nucleus has size 0
    index = np.logical_and(normalized['label'] == 2, normalized['fov'] == 'Point0')
    assert normalized.loc[index, 'cell_size_nuclear'].values == 0
