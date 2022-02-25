import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
import xarray as xr

from ark.mibi.mibitracker_utils import MibiRequests
import ark.mibi.qc_comp as qc_comp
import ark.utils.io_utils as io_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

import os
import pytest
import tempfile


RUN_POINT_NAMES = ['Point%d' % i for i in range(1, 13)]
RUN_POINT_IDS = list(range(661, 673))

# NOTE: all fovs and all channels will be tested in the example_qc_metric_eval notebook test
FOVS_CHANS_TEST_MIBI = [
    (None, ['CCL8', 'CD11b'], None, RUN_POINT_NAMES, RUN_POINT_IDS),
    (None, ['CCL8', 'CD11b'], "TIFs", RUN_POINT_NAMES, RUN_POINT_IDS),
    (['Point1'], None, None, RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], None, "TIFs", RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], ['CCL8', 'CD11b'], None, RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], ['CCL8', 'CD11b'], "TIFs", RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1])
]


FOVS_CHANS_TEST_QC = [
    (None, None, False),
    (None, None, True),
    (['fov0', 'fov1'], None, False),
    (['fov0', 'fov1'], None, True),
    (None, ['chan0', 'chan1'], False),
    (None, ['chan0', 'chan1'], True),
    (['fov0', 'fov1'], ['chan0', 'chan1'], False),
    (['fov0', 'fov1'], ['chan0', 'chan1'], True)
]

MIBITRACKER_EMAIL = 'qc.mibi@gmail.com'
MIBITRACKER_PASSWORD = 'The_MIBI_Is_Down_Again1!?'
MIBITRACKER_RUN_NAME = '191008_JG85b'
MIBITRACKER_RUN_LABEL = 'JG85_Run2'


def test_create_mibitracker_request_helper():
    # error check: bad email and/or password provided
    mr = qc_comp.create_mibitracker_request_helper('bad_email', 'bad_password')
    assert mr is None

    # test creation works (just test the correct type returned)
    mr = qc_comp.create_mibitracker_request_helper(MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD)
    assert type(mr) == MibiRequests


@pytest.mark.parametrize(
    "test_fovs,test_chans,test_sub_folder,actual_points,actual_ids",
    FOVS_CHANS_TEST_MIBI
)
def test_download_mibitracker_data(test_fovs, test_chans, test_sub_folder,
                                   actual_points, actual_ids):
    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad base_dir provided
        with pytest.raises(FileNotFoundError):
            qc_comp.download_mibitracker_data('', '', '', '', 'bad_base_dir', '', '')

        # error check: bad run_name and/or run_label provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD, 'bad_run_name', 'bad_run_label',
                temp_dir, '', ''
            )

        # bad fovs provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, '', '', fovs=['Point0', 'Point1']
            )

        # bad channels provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, '', '', channels=['B', 'C']
            )

        # ensure test to remove tiff_dir if it already exists runs
        os.mkdir(os.path.join(temp_dir, 'sample_tiff_dir'))

        # error check: tiff_dir that already exists provided with overwrite_tiff_dir=False
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, 'sample_tiff_dir', overwrite_tiff_dir=False,
                img_sub_folder=test_sub_folder, fovs=test_fovs, channels=test_chans
            )

        # run the data
        run_order = qc_comp.download_mibitracker_data(
            MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
            MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
            temp_dir, 'sample_tiff_dir', overwrite_tiff_dir=True,
            img_sub_folder=test_sub_folder, fovs=test_fovs, channels=test_chans
        )

        # for testing purposes, set test_fovs and test_chans to all fovs and channels
        # if they're set to None
        if test_fovs is None:
            test_fovs = ['Point%d' % i for i in np.arange(1, 13)]

        if test_chans is None:
            test_chans = [
                'CD115', 'C', 'Au', 'CCL8', 'CD11c', 'Ca', 'Background',
                'CD11b', 'CD192', 'CD19', 'CD206', 'CD25', 'CD4', 'CD45.1',
                'CD3', 'CD31', 'CD49b', 'CD68', 'CD45.2', 'FceRI', 'DNA', 'CD8',
                'F4-80', 'Fe', 'IL-1B', 'Ly-6C', 'FRB', 'Lyve1', 'Ly-6G', 'MHCII',
                'Na', 'Si', 'SMA', 'P', 'Ta', 'TREM2'
            ]

        # set the sub folder to a blank string if None
        if test_sub_folder is None:
            test_sub_folder = ""

        # get the contents of tiff_dir
        tiff_dir_contents = os.listdir(os.path.join(temp_dir, 'sample_tiff_dir'))

        # assert all the fovs are contained in the dir
        tiff_dir_fovs = [d for d in tiff_dir_contents if
                         os.path.isdir(os.path.join(temp_dir, 'sample_tiff_dir', d))]
        misc_utils.verify_same_elements(
            created_fov_dirs=tiff_dir_fovs,
            provided_fov_dirs=test_fovs
        )

        # assert for each fov the channels created are correct
        for fov in tiff_dir_fovs:
            # list all the files in the fov folder (and sub folder)
            # remove file extensions so raw channel names are extracted
            channel_files = io_utils.remove_file_extensions(os.listdir(
                os.path.join(temp_dir, 'sample_tiff_dir', fov, test_sub_folder)
            ))

            # assert the channel names are the same
            misc_utils.verify_same_elements(
                create_channels=channel_files,
                provided_channels=test_chans
            )

        # assert that the run order created is correct for both points and ids
        run_fov_names = [ro[0] for ro in run_order]
        run_fov_ids = [ro[1] for ro in run_order]

        assert run_fov_names == actual_points
        assert run_fov_ids == actual_ids


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


@pytest.mark.parametrize("test_fovs,test_chans,test_gaussian_blur", FOVS_CHANS_TEST_QC)
def test_compute_qc_metrics(test_fovs, test_chans, test_gaussian_blur):
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
                tiff_dir, img_sub_folder, channels=['bad_chan']
            )

        # test sets of fovs and channels and Gaussian blur turned on or off
        # NOTE: leave default Gaussian blur sigma at 1 (same test regardless of sigma)
        qc_data = qc_comp.compute_qc_metrics(
            tiff_dir, img_sub_folder, fovs=test_fovs, channels=test_chans,
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
