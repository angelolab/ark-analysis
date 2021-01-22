import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def mocked_cluster_pixels(base_dir, chan_list):
    # assign marker names and create sample marker information
    chan_list = ['Marker1', 'Marker2', 'Marker3']
    sample_matrix = pd.DataFrame(np.random.rand(100, 3), columns=chan_list)

    # assign random cluster IDs
    sample_matrix['ClusterID'] = np.random.randint(low=0, high=100, size=100)

    # write the final data to CSV
    sample_matrix.to_csv(os.path.join(base_dir, 'pixel_mat_clustered.csv'), index=False)


def test_create_pixel_matrix():
    # must be float64, since that's how Candace's data comes in
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=['fov0', 'fov1'],
                                                  channel_names=['chan0', 'chan1'],
                                                  dtype='float64')

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=['fov0', 'fov1'],
                                                  compartment_names=['whole_cell'])

    # pass invalid fov names
    with pytest.raises(ValueError):
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, fovs=['fov1', 'fov2'])

    # pass invalid channel names
    with pytest.raises(ValueError):
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, channels=['chan1', 'chan2'])

    # test on all fovs and channels
    flowsom_data = som_utils.create_pixel_matrix(sample_img_xr, sample_labels)

    # assert all fovs and channels are covered
    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov0', 'fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0', 'chan1'])

    # assert no rows sum to 0
    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of fovs
    flowsom_data = som_utils.create_pixel_matrix(sample_img_xr, sample_labels, fovs=['fov1'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0', 'chan1'])

    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of channels
    flowsom_data = som_utils.create_pixel_matrix(sample_img_xr, sample_labels, channels=['chan0'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov0', 'fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0'])

    # test on a subset of fovs and channels
    flowsom_data = som_utils.create_pixel_matrix(sample_img_xr, sample_labels,
                                                 fovs=['fov1'], channels=['chan0'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0'])

    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)


def test_cluster_pixels(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        # add mocked function to create a dummy pixel matrix with cluster labels
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # create a sample preprocessed pixel matrix with specified channel names
        chan_list = ['Marker1', 'Marker2', 'Marker3']
        sample_matrix = pd.DataFrame(np.random.rand(100, 3), columns=chan_list)
        sample_matrix.to_csv(os.path.join(temp_dir, 'example_pixel_matrix.csv'))

        # run clustering using mocked function
        som_utils.cluster_pixels(temp_dir, chan_list)

        # assert the clustered file has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_clustered.csv'))
