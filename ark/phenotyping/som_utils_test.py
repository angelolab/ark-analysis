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
    sample_pixel_mat = pd.read_csv(os.path.join(base_dir, 'pixel_mat_preprocessed.csv'),
                                   usecols=chan_list)

    sample_pixel_mat = sample_pixel_mat

    # we will assign cluster labels based on the average of the rows
    sample_pixel_mat['Cluster_ID'] = sample_pixel_mat.mean(axis=1)

    # multiply by 100 and truncate for an integer cluster assignment
    sample_pixel_mat['Cluster_ID'] *= 100
    sample_pixel_mat['Cluster_ID'] = sample_pixel_mat['Cluster_ID'].astype(int)

    # write the final data to CSV
    sample_pixel_mat.to_csv(os.path.join(base_dir, 'pixel_mat_clustered.csv'), index=False)


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
    # basic error check: bad path to preprocessed matrix
    with pytest.raises(FileNotFoundError):
        som_utils.cluster_pixels(['Marker1', 'Marker2'], 'bad_base_dir')

    with tempfile.TemporaryDirectory() as temp_dir:
        # add mocked function to create a dummy pixel matrix with cluster labels
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # create a sample preprocessed pixel matrix with specified channel names
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        sample_matrix = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
        sample_matrix.to_csv(os.path.join(temp_dir, 'pixel_mat_preprocessed.csv'))

        # subset specific markers for clustering
        chan_select = ['Marker1', 'Marker2']

        # run "clustering" using mocked function
        som_utils.cluster_pixels(temp_dir, chan_select)

        # assert the clustered file has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_clustered.csv'))

        sample_pixel_clustered = pd.read_csv(os.path.join(temp_dir, 'pixel_mat_clustered.csv'))

        # assert the columns we subsetted are the same (aside from the Cluster_ID column)
        misc_utils.verify_same_elements(
            selected_markers=chan_select,
            pixel_mat_cols=sample_pixel_clustered.drop(columns='Cluster_ID').columns.values)

        # assert we didn't assign any cluster 100 or above
        cluster_ids = sample_pixel_clustered['Cluster_ID'].values
        assert np.all(cluster_ids < 100)
