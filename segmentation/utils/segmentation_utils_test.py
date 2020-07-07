import numpy as np
import xarray as xr
import os
import pytest
import tempfile

import skimage.morphology as morph
from skimage.morphology import erosion

from segmentation.utils import segmentation_utils


def _generate_deepcell_ouput(fov_num=2):
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


# TODO: add better test data for the actual local maxima/thresholding functionality
def test_watershed_transform():
    model_output = _generate_deepcell_ouput()
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

    # TODO: Right now these tests are copied from above, can think about refactoring
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


def test_generate_expression_matrix():
    # create input data
    cell_mask = np.zeros((40, 40), dtype='int16')
    cell_mask[4:10, 4:8] = 1
    cell_mask[15:25, 20:30] = 2
    cell_mask[30:32, 3:28] = 3
    cell_mask[35:40, 15:18] = 4

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

    normalized, transformed = segmentation_utils.generate_expression_matrix(segmentation_masks,
                                                                            channel_data)

    assert normalized.shape[0] == 7

    assert np.all(normalized['chan0'] == np.repeat(1, len(normalized)))
    assert np.all(normalized['chan1'] == np.repeat(5, len(normalized)))
    assert np.all(normalized['chan2'] == normalized['chan2'])
