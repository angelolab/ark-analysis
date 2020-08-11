import numpy as np
import xarray as xr

from segmentation.utils import signal_extraction


def test_positive_pixels_extraction():
    # this function tests the functionality of positive pixels extraction
    # where we count the number of non-zero values for each channel

    # create a series of constant arrays
    base_channel = np.zeros((10, 10))

    combined_channels = np.stack((base_channel, base_channel + 1, base_channel + 2), axis=-1)
    combined_channels = xr.DataArray(combined_channels)

    coords = np.stack((np.arange(4), np.arange(4)), axis=-1)

    channel_counts = signal_extraction.positive_pixels_extraction(cell_coords=coords,
                                                                  image_data=combined_channels)

    assert np.all(channel_counts == [0, 4, 4])


def test_center_weighting_extraction():
    # this function tests the functionality of center weighting extraction
    # where we add a weighting scheme with more confidence toward the center
    # before summing across each channel

    # create a series of constant arrays
    base_channel = np.ones((10, 10))

    combined_channels = np.stack((base_channel, base_channel * 2, base_channel * 4), axis=-1)
    combined_channels = xr.DataArray(combined_channels)

    coords = np.stack((np.arange(6), np.arange(6)), axis=-1)

    channel_counts = signal_extraction.center_weighting_extraction(cell_coords=coords,
                                                                   image_data=combined_channels)

    assert np.all(channel_counts == [3.5, 7., 14.])


def test_default_extraction():
    # this function tests the functionality of default weighting extraction
    # where we just sum across each channel

    # create a series of constant arrays
    base_channel = np.ones((10, 10))

    combined_channels = np.stack((base_channel, base_channel * 2, base_channel * 5), axis=-1)
    combined_channels = xr.DataArray(combined_channels)

    coords = np.stack((np.arange(4), np.arange(4)), axis=-1)

    channel_counts = signal_extraction.default_extraction(cell_coords=coords,
                                                          image_data=combined_channels)

    assert np.all(channel_counts == [4, 8, 20])
