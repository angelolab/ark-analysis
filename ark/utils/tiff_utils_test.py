import os
import tempfile

import numpy as np
import pytest

from ark.utils import test_utils, tiff_utils


# test read_mibitiff on static tiff file
# shouldn't use test_utils here since it uses write_mibitiff
def test_read_mibitiff():
    # should throw error on standard tif load
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepaths, _ = test_utils.create_paired_xarray_fovs(temp_dir, ['test_fov'], ['chan0'])

            # should raise value error
            img_data, all_channels = tiff_utils.read_mibitiff(filepaths['test_fov'][0] + '.tiff')

    # should throw error on loading bad channel!
    with pytest.raises(IndexError):
        with tempfile.TemporaryDirectory() as temp_dir:
            filepaths, _ = test_utils.create_paired_xarray_fovs(
                temp_dir, ['test_fov'], ['chan0', 'chan1'], mode='mibitiff'
            )

            img_data, all_channels = tiff_utils.read_mibitiff(
                filepaths['test_fov'], channels=['chan2']
            )

    fovs, chans = test_utils.gen_fov_chan_names(1, 10)
    with tempfile.TemporaryDirectory() as temp_dir:
        # create dummy data to load
        filepaths, true_data = test_utils.create_paired_xarray_fovs(temp_dir, fov_names=fovs,
                                                                    img_shape=(1024, 1024),
                                                                    channel_names=chans,
                                                                    mode='mibitiff', fills=True)

        # load in all the channels
        img_data, chan_tups = tiff_utils.read_mibitiff(filepaths[fovs[0]])
        assert img_data.shape == (1024, 1024, len(chan_tups))

        # load in a subset of channels
        chan_names = [chan_tup[1] for chan_tup in chan_tups]
        subset_img_data, subset_chan = tiff_utils.read_mibitiff(filepaths[fovs[0]],
                                                                channels=chan_names[:3])

        assert subset_chan == chan_tups[:3]
        assert np.all(img_data[:, :, :3] == subset_img_data)


# test write_mibitiff and verify with read_mibitiff
# test utils uses write_mibitiff so we can just use that to test it
def test_write_mibitiff():
    fovs, chans = test_utils.gen_fov_chan_names(1, 10)
    with tempfile.TemporaryDirectory() as temp_dir:
        # this uses write_mibitiff
        filepaths, true_data = test_utils.create_paired_xarray_fovs(temp_dir, fov_names=fovs,
                                                                    channel_names=chans,
                                                                    mode='mibitiff', fills=True)

        # validate correct tiff writeout via read_mibitiff
        load_data, chan_tups = tiff_utils.read_mibitiff(filepaths[fovs[0]])

        assert np.all(true_data[0, :, :, :].values == load_data)
