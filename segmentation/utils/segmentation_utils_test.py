import numpy as np
import xarray as xr
import os

from segmentation.utils import segmentation_utils


def test_watershed_transform():
    # load data
    base_dir = "segmentation/tests/test_output_files/segmentation_output/"
    pixel = xr.open_dataarray(base_dir + "test_input_pixel_processed.nc")
    watershed = xr.open_dataarray(base_dir + "test_input_watershed_processed.nc")
    input = xr.open_dataarray(base_dir + "test_input_channel.nc")

    segmentation_utils.watershed_transform(pixel_xr=pixel, watershed_xr=watershed, channel_xr=input,
                                           overlay_channels=["Na"], output_dir=base_dir, watershed_maxs=True,
                                           )
    # clean up output files
    temp_files = os.listdir(base_dir)
    temp_files = [file for file in temp_files if "test_input" not in file]

    for file in temp_files:
        os.remove(base_dir + file)


# testing for segmentation utils
def test_segment_images():
    # first create cell masks
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

    cell_xr = xr.DataArray(np.expand_dims(cell_mask, axis=-1), coords=[range(40), range(40), ["cell_mask"]],
                           dims=["rows", "cols", "subcell_loc"])

    channel_xr = xr.DataArray(channel_data, coords=[range(40), range(40), ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                                    dims=["rows", "cols", "channels"])

    segmentation_output = segmentation_utils.segment_images(channel_xr, cell_xr)

    # check that channel 0 counts are same as cell size
    assert np.array_equal(segmentation_output.loc["cell_mask", :, "cell_size"].values,
                        segmentation_output.loc["cell_mask", :, "chan0"].values)

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(segmentation_output.loc["cell_mask", :, "cell_size"].values * 5,
                          segmentation_output.loc["cell_mask", :, "chan1"].values)

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(segmentation_output.loc["cell_mask", :, "chan2"].values,
                          segmentation_output.loc["cell_mask", :, "chan1"].values)

    # check that only cell1 is negative for channel 3
    assert segmentation_output.loc["cell_mask", :, "chan3"][1] == 0
    assert np.all(segmentation_output.loc["cell_mask", :, "chan3"][2:] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output.loc["cell_mask", :, "chan4"][2] > 0
    assert np.all(segmentation_output.loc["cell_mask", :, "chan4"][:2] == 0)
    assert np.all(segmentation_output.loc["cell_mask", :, "chan4"][3:] == 0)

    # check that cell sizes are correct
    sizes = np.sum(cell_mask == -1), np.sum(cell_mask == 1), np.sum(cell_mask == 2), np.sum(cell_mask == 3)
    assert np.array_equal(sizes, segmentation_output.loc["cell_mask", :3, "cell_size"])


def test_extract_single_cell_data():
    base_dir = "segmentation/tests/test_output_files/single_cell_data/"

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

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

    # generate data for two points offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask
    cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data
    channel_datas[1, 5:, 5:, :] = channel_data[:-5, :-5]

    segmentation_masks = xr.DataArray(cell_masks,
                                      coords=[["Point1", "Point2"], range(40), range(40), ["segmentation_label"]],
                                      dims=["points", "rows", "cols", "channels"])

    channel_data = xr.DataArray(channel_datas,
                                coords=[["Point1", "Point2"], range(40), range(40),
                                        ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                dims=["points", "rows", "cols", "channels"])

    segmentation_utils.extract_single_cell_data(segmentation_masks, channel_data, base_dir)

    temp_files = os.listdir(base_dir)
    temp_files = [file for file in temp_files if "Point" in file]

    for file in temp_files:
        os.remove(base_dir + file)

    os.rmdir(base_dir)

