import numpy as np
import xarray as xr

from segmentation.utils import signal_extraction


def test_default_extraction():
    # create a series of constant arrays
    base_channel = np.ones((10, 10))

    combined_channels = np.stack((base_channel, base_channel * 2, base_channel * 5), axis=-1)
    combined_channels = xr.DataArray(combined_channels)

    coords = np.array([[0, 0],
                       [1, 1],
                       [2, 2],
                       [3, 3]])

    channel_counts = signal_extraction.default_extraction(cell_coords=coords,
                                                          image_data=combined_channels)

    assert np.all(channel_counts == [4, 8, 20])
