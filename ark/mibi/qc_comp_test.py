import numpy as np
import pandas as pd
import xarray as xr

import ark.mibi.qc_comp as qc_comp
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

import os
import pytest
import tempfile


FOVS_CHANS_NON_MIBITIFF = [
    (['fov0', 'fov1', 'fov2'], ['chan0', 'chan1', 'chan2']),
    (['fov0', 'fov1'], ['chan0', 'chan1', 'chan2']),
    (['fov0', 'fov1', 'fov2'], ['chan0', 'chan1']),
    (['fov0', 'fov1'], ['chan0', 'chan1'])
]


def test_compute_nonzero_mean_intensity():
    sample_img_arr = xr.DataArray(
        np.random.randint(0, 2, (3, 1024, 1024, 3)),
        coords=[['fov0', 'fov1', 'fov2'],
                np.arange(1024),
                np.arange(1024),
                ['chan0', 'chan1', 'chan2']],
        dims=['fovs', 'x', 'y', 'channels']
    )

    sample_nonzero_mean = qc_comp.compute_nonzero_mean_intensity(sample_img_arr)

    # output shape must match num_fovs x num_channels
    assert sample_nonzero_mean.shape == (3, 3)

    # all nonzero means have to be greater than 0
    assert np.all(sample_nonzero_mean > 0)


def test_compute_total_intensity():
    sample_img_arr = xr.DataArray(
        np.random.randint(0, 2, (3, 1024, 1024, 3)),
        coords=[['fov0', 'fov1', 'fov2'],
                np.arange(1024),
                np.arange(1024),
                ['chan0', 'chan1', 'chan2']],
        dims=['fovs', 'x', 'y', 'channels']
    )

    sample_total_intensity = qc_comp.compute_total_intensity(sample_img_arr)

    # output shape must match num_fovs x num_channels
    assert sample_total_intensity.shape == (3, 3)

    # all total intensities have to be greater than 0 but also less than or equal to 1024 * 1024
    assert np.all(
        np.logical_and(sample_total_intensity > 0, sample_total_intensity <= 1024 * 1024)
    )


def test_compute_99_9_intensity():
    sample_img_arr = xr.DataArray(
        np.random.randint(0, 2, (3, 1024, 1024, 3)),
        coords=[['fov0', 'fov1', 'fov2'],
                np.arange(1024),
                np.arange(1024),
                ['chan0', 'chan1', 'chan2']],
        dims=['fovs', 'x', 'y', 'channels']
    )

    sample_99_9_intensity = qc_comp.compute_99_9_intensity(sample_img_arr)

    # output shape must match num_fovs x num_channels
    assert sample_99_9_intensity.shape == (3, 3)

    # all 99.9% intensities have to be greater than or equal to 0 but also less than or equal to 1
    assert np.all(
        np.logical_and(sample_99_9_intensity >= 0, sample_99_9_intensity <= 1)
    )


@pytest.mark.parametrize("test_fovs,test_chans", FOVS_CHANS_NON_MIBITIFF)
def test_compute_qc_metrics_non_mibitiff(test_fovs, test_chans):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 fovs and 3 channels
        fovs, chans = test_utils.gen_fov_chan_names(3, 3)

        # make the sample data
        tiff_dir = os.path.join(temp_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=chans,
            img_shape=(40, 40),
            sub_dir=img_sub_folder,
            fills=True,
            dtype="int16"
        )

        # invalid channels provided
        with pytest.raises(ValueError):
            qc_comp.compute_qc_metrics(
                tiff_dir, img_sub_folder, chans=['bad_chan']
            )

        # test sets of fovs and channels
        nonzero_mean, total_intensity, intensity_99_9 = qc_comp.compute_qc_metrics(
            tiff_dir, img_sub_folder, fovs=test_fovs, chans=test_chans
        )

        # assert fovs are correct
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            nzm_fovs=nonzero_mean['fovs'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            ti_fovs=total_intensity['fovs'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            i99_9_fovs=intensity_99_9['fovs'].values
        )

        # assert channels are correct
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            nzm_chans=nonzero_mean.drop(columns='fovs').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            nzm_chans=total_intensity.drop(columns='fovs').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            nzm_chans=intensity_99_9.drop(columns='fovs').columns.values
        )
