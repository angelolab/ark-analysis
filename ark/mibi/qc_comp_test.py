import numpy as np
import pandas as pd
import xarray as xr

import ark.mibi.qc_comp as qc_comp
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

import os
import pytest
import tempfile


FOVS_CHANS_TEST = [
    (None, None, False),
    (None, None, True),
    (['fov0', 'fov1'], None, False),
    (['fov0', 'fov1'], None, True),
    (None, ['chan0', 'chan1'], False),
    (None, ['chan0', 'chan1'], True),
    (['fov0', 'fov1'], ['chan0', 'chan1'], False),
    (['fov0', 'fov1'], ['chan0', 'chan1'], True)
]


def test_compute_nonzero_mean_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_nonzero_mean = qc_comp.compute_nonzero_mean_intensity(sample_img_arr)
    assert sample_nonzero_mean == 3


def test_compute_total_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_total_intensity = qc_comp.compute_total_intensity(sample_img_arr)
    assert sample_total_intensity == 15


def test_compute_99_9_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_99_9_intensity = qc_comp.compute_99_9_intensity(sample_img_arr)
    assert np.allclose(sample_99_9_intensity, 5, rtol=1e-02)


def test_compute_qc_metrics_batch():
    # define the fovs and chans for this test batch
    fovs = ['fov0', 'fov1', 'fov2']
    chans = ['chan0', 'chan1', 'chan2']

    # create a test batched image array
    sample_img_arr = xr.DataArray(
        np.random.rand(3, 10, 10, 3),
        coords=[fovs, np.arange(10), np.arange(10), chans],
        dims=['fov', 'x', 'y', 'channel']
    )

    # test with Gaussian blurring turned on and off
    for gaussian_blur in [False, True]:
        qc_data_batch = qc_comp.compute_qc_metrics_batch(
            sample_img_arr, fovs, chans, gaussian_blur=gaussian_blur
        )

        # extract the QC metric batch data separately
        nonzero_mean_batch = qc_data_batch['nonzero_mean_batch']
        total_intensity_batch = qc_data_batch['total_intensity_batch']
        intensity_99_9_batch = qc_data_batch['99_9_intensity_batch']

        # assert the fovs are correct
        misc_utils.verify_same_elements(
            provided_fovs=fovs,
            nzm_fovs=nonzero_mean_batch['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=fovs,
            ti_fovs=total_intensity_batch['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=fovs,
            i99_9_fovs=intensity_99_9_batch['fov'].values
        )

        # assert the chans are correct
        misc_utils.verify_same_elements(
            provided_chans=chans,
            nzm_chans=nonzero_mean_batch.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=chans,
            nzm_chans=total_intensity_batch.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=chans,
            nzm_chans=intensity_99_9_batch.drop(columns='fov').columns.values
        )


@pytest.mark.parametrize("test_fovs,test_chans,test_gaussian_blur", FOVS_CHANS_TEST)
def test_compute_qc_metrics_mibitiff(test_fovs, test_chans, test_gaussian_blur):
    # is_mibitiff True case, load from mibitiff file structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 fovs and 2 mibitiff_imgs
        fovs, chans = test_utils.gen_fov_chan_names(3, 2)

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # define a subset of fovs with file extensions
        fovs_subset_ext = fovs[:2]
        fovs_subset_ext[0] = str(fovs_subset_ext[0]) + ".tif"
        fovs_subset_ext[1] = str(fovs_subset_ext[1]) + ".tiff"

        tiff_dir = os.path.join(temp_dir, "mibitiff_inputs")

        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=chans,
            img_shape=(40, 40),
            mode='mibitiff',
            dtype=np.float32
        )

        # invalid channels provided
        with pytest.raises(ValueError):
            qc_comp.compute_qc_metrics(
                tiff_dir, is_mibitiff=True, chans=['bad_chan']
            )

        # test sets of fovs and channels
        qc_data = qc_comp.compute_qc_metrics(
            tiff_dir, is_mibitiff=True, fovs=test_fovs, chans=test_chans,
            gaussian_blur=test_gaussian_blur
        )

        # extract the QC metric data separately
        nonzero_mean = qc_data['nonzero_mean']
        total_intensity = qc_data['total_intensity']
        intensity_99_9 = qc_data['99_9_intensity']

        # assert fovs are correct (if fovs is None, set to all fovs)
        if test_fovs is None:
            test_fovs = fovs

        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            nzm_fovs=nonzero_mean['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            ti_fovs=total_intensity['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            i99_9_fovs=intensity_99_9['fov'].values
        )

        # assert channels are correct (if chans is None, set to all chans)
        if test_chans is None:
            test_chans = chans

        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            nzm_chans=nonzero_mean.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            ti_chans=total_intensity.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            i99_9_chans=intensity_99_9.drop(columns='fov').columns.values
        )


@pytest.mark.parametrize("test_fovs,test_chans,test_gaussian_blur", FOVS_CHANS_TEST)
def test_compute_qc_metrics_non_mibitiff(test_fovs, test_chans, test_gaussian_blur):
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

        # test sets of fovs and channels and Gaussian blur turned on or off
        # NOTE: leave default Gaussian blur sigma at 1 (same test regardless of sigma)
        qc_data = qc_comp.compute_qc_metrics(
            tiff_dir, img_sub_folder, fovs=test_fovs, chans=test_chans,
            gaussian_blur=test_gaussian_blur
        )

        nonzero_mean = qc_data['nonzero_mean']
        total_intensity = qc_data['total_intensity']
        intensity_99_9 = qc_data['99_9_intensity']

        # assert fovs are correct (if fovs is None, set to all fovs)
        if test_fovs is None:
            test_fovs = fovs

        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            nzm_fovs=nonzero_mean['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            ti_fovs=total_intensity['fov'].values
        )
        misc_utils.verify_same_elements(
            provided_fovs=test_fovs,
            i99_9_fovs=intensity_99_9['fov'].values
        )

        # assert channels are correct (if chans is None, set to all chans)
        if test_chans is None:
            test_chans = chans

        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            nzm_chans=nonzero_mean.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            ti_chans=total_intensity.drop(columns='fov').columns.values
        )
        misc_utils.verify_same_elements(
            provided_chans=test_chans,
            i99_9_chans=intensity_99_9.drop(columns='fov').columns.values
        )


def test_visualize_qc_metrics():
    # define the channels to use
    chans = ['chan0', 'chan1', 'chan2']

    # define the fov names to use for each channel
    fov_batches = [['fov0', 'fov1'], ['fov2', 'fov3'], ['fov4', 'fov5']]

    # define the test melted DataFrame for an arbitrary QC metric
    sample_qc_metric_data = pd.DataFrame()

    # for each channel append a random set of data for each fov associated with the QC metric
    for chan, fovs in zip(chans, fov_batches):
        chan_data = pd.DataFrame(np.random.rand(len(fovs)), columns=['sample_qc_metric'])
        chan_data['fov'] = fovs
        chan_data['channel'] = chan

        sample_qc_metric_data = pd.concat([sample_qc_metric_data, chan_data])

    with tempfile.TemporaryDirectory() as temp_dir:
        # test without saving
        qc_comp.visualize_qc_metrics(sample_qc_metric_data, 'sample_qc_metric')
        assert not os.path.exists(os.path.join(temp_dir, 'sample_qc_metric_barplot_stats.png'))

        # test with saving
        qc_comp.visualize_qc_metrics(sample_qc_metric_data, 'sample_qc_metric', save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'sample_qc_metric_barplot_stats.png'))
