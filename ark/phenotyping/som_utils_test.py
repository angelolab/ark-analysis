import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def mocked_cluster_pixels(fovs, channels, base_dir,
                          pixel_pre_name='pixel_mat_preprocessed.hdf5',
                          pixel_subset_name='pixel_mat_subsetted.csv',
                          pixel_cluster_name='pixel_mat_clustered.hdf5'):
    # read the subsetted pixel matrix
    pixel_mat_sub = pd.read_csv(os.path.join(base_dir, pixel_subset_name))

    # get the means across the rows using only the columns we need
    sub_means = pixel_mat_sub[channels].mean(axis=1)

    # multiply by 100 and truncate to int to get an actual cluster id
    cluster_ids = sub_means * 100
    cluster_ids = cluster_ids.astype(int)

    # we will now broadcast these cluster_ids across each DataFrame in pixel_mat_preprocessed.hdf5
    for fov in fovs:
        # get the fovs pixel matrix
        flowsom_clustered = pd.read_hdf(os.path.join(base_dir, pixel_pre_name), key=fov)

        # repeat the same cluster id 10 times for each submatrix
        flowsom_clustered['Cluster'] = cluster_ids.repeat(5).reset_index(drop=True)

        # write clustered data to HDF5
        flowsom_clustered.to_hdf(os.path.join(base_dir, pixel_cluster_name), key=fov, mode='a')


def test_create_pixel_matrix():
    fovs = ['fov0', 'fov1']
    chans = ['chan0', 'chan1']

    # must be float64, since that's how Candace's data comes in
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=fovs,
                                                  channel_names=chans,
                                                  dtype='float64')

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=['fov0', 'fov1'],
                                                  compartment_names=['whole_cell'])

    # basic error checking
    with tempfile.TemporaryDirectory() as temp_dir:
        # pass invalid fov names
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          fovs=['fov1', 'fov2'])

        # pass invalid channel names
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          channels=['chan1', 'chan2'])

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, 'bad_base_dir')

    # test on all fovs and channels, use default file name
    with tempfile.TemporaryDirectory() as temp_dir:
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir)

        # check that we actually created a preprocessed HDF5
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'))

        # read in this way so we get info about the keys (fovs)
        flowsom_pre = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), 'r')

        # make sure the fovs match
        misc_utils.verify_same_elements(
            flowsom_fovs=[fov.replace('/', '') for fov in flowsom_pre.keys()],
            provided_fovs=fovs)

        # assert all fovs and channels are covered
        for fov in flowsom_pre.keys():
            # get the data for the specific fov
            flowsom_pre_fov = flowsom_pre.get(fov)

            # assert the channel names are the same
            misc_utils.verify_same_elements(flowsom_chans=flowsom_pre_fov.columns.values[:-4],
                                            provided_chans=chans)

            # assert no rows sum to 0
            assert np.all(flowsom_pre_fov.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of fovs
    with tempfile.TemporaryDirectory() as temp_dir:
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir, fovs=['fov1'])

        # check that we actually created a preprocessed HDF5
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'))

        # read in this way so we get info about the keys (fovs)
        flowsom_pre = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), 'r')

        # make sure the fovs match
        misc_utils.verify_same_elements(
            flowsom_fovs=[fov.replace('/', '') for fov in flowsom_pre.keys()],
            provided_fovs=['fov1'])

        # assert all fovs and channels are covered
        for fov in flowsom_pre.keys():
            # get the data for the specific fov
            flowsom_pre_fov = flowsom_pre.get(fov)

            # assert the channel names are the same
            misc_utils.verify_same_elements(flowsom_chans=flowsom_pre_fov.columns.values[:-4],
                                            provided_chans=chans)

            # assert no rows sum to 0
            assert np.all(flowsom_pre_fov.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of channels
    with tempfile.TemporaryDirectory() as temp_dir:
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir, channels=['chan0'])

        # check that we actually created a preprocessed HDF5
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'))

        # read in this way so we get info about the keys (fovs)
        flowsom_pre = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), 'r')

        # make sure the fovs match
        misc_utils.verify_same_elements(
            flowsom_fovs=[fov.replace('/', '') for fov in flowsom_pre.keys()],
            provided_fovs=fovs)

        # assert all fovs and channels are covered
        for fov in flowsom_pre.keys():
            # get the data for the specific fov
            flowsom_pre_fov = flowsom_pre.get(fov)

            # assert the channel names are the same
            misc_utils.verify_same_elements(flowsom_chans=flowsom_pre_fov.columns.values[:-4],
                                            provided_chans=['chan0'])

            # assert no rows sum to 0
            assert np.all(flowsom_pre_fov.loc[:, ['chan0']].sum(axis=1).values != 0)

    # test on a subset of fovs and channels
    with tempfile.TemporaryDirectory() as temp_dir:
        som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                      fovs=['fov0'], channels=['chan0'])

        # check that we actually created a preprocessed HDF5
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'))

        # read in this way so we get info about the keys (fovs)
        flowsom_pre = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), 'r')

        # make sure the fovs match
        misc_utils.verify_same_elements(
            flowsom_fovs=[fov.replace('/', '') for fov in flowsom_pre.keys()],
            provided_fovs=['fov0'])

        # assert all fovs and channels are covered
        for fov in flowsom_pre.keys():
            # get the data for the specific fov
            flowsom_pre_fov = flowsom_pre.get(fov)

            # assert the channel names are the same
            misc_utils.verify_same_elements(flowsom_chans=flowsom_pre_fov.columns.values[:-4],
                                            provided_chans=['chan0'])

            # assert no rows sum to 0
            assert np.all(flowsom_pre_fov.loc[:, ['chan0']].sum(axis=1).values != 0)


def test_subset_pixels():
    # create two dummy sample dataframes
    sample_df_0 = pd.DataFrame(np.random.rand(100, 6),
                               columns=['chan0', 'chan1', 'fov',
                                        'row_index', 'col_index', 'segmentation_label'])
    sample_df_1 = pd.DataFrame(np.random.rand(100, 6),
                               columns=['chan0', 'chan1', 'fov',
                                        'row_index', 'col_index', 'segmentation_label'])

    # standardize the fov columns, this will help with testing
    sample_df_0['fov'] = 'fov0'
    sample_df_1['fov'] = 'fov1'

    # set the list of fovs we'll be using
    fovs = ['fov0', 'fov1']

    with tempfile.TemporaryDirectory() as temp_dir:
        # write the dataframes to HDF5
        sample_df_0.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), key='fov0', mode='a')
        sample_df_1.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), key='fov1', mode='a')

        # bad subset percentage set
        with pytest.raises(ValueError):
            som_utils.subset_pixels(fovs, temp_dir, subset_percent=1.1)

        # bad preprocessed HDF5 file path set
        with pytest.raises(FileNotFoundError):
            som_utils.subset_pixels(fovs, temp_dir, hdf_name='bad_preprocessed_path.hdf5')

        # now actually subset pixels with default settings
        som_utils.subset_pixels(fovs, temp_dir)

        # assert we actually created the subsetted CSV
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted.csv'))

        flowsom_subset = pd.read_csv(os.path.join(temp_dir, 'pixel_mat_subsetted.csv'))

        # assert that the fovs in the fov column match up
        misc_utils.verify_same_elements(flowsom_fovs=flowsom_subset['fov'].unique(),
                                        provided_fovs=fovs)

        # for each fov, assert that we selected 10 elements (default subset_percent is 0.1)
        for fov in fovs:
            flowsom_subset_fov = flowsom_subset[flowsom_subset['fov'] == fov]

            assert flowsom_subset_fov.shape[0] == 10


def test_cluster_pixels(mocker):
    # basic error checks: bad path to preprocessed and subsetted matrices
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], channels=['Marker1'],
                                     base_dir=temp_dir, pixel_pre_name='bad_path.hdf5')

        # write a random HDF5 for the undefined subsetted matrix test
        random_df = pd.DataFrame(np.random.rand(10, 2))
        random_df.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), key='fov0', mode='a')

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], channels=['Marker1'],
                                     base_dir=temp_dir, pixel_subset_name='bad_path.csv')

    with tempfile.TemporaryDirectory() as temp_dir:
        # add mocked function to create a dummy pixel matrix with cluster labels
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'col_index', 'segmentation_label']

        # create the subsetted DataFrame to write
        pixel_mat_subsetted = pd.DataFrame(columns=colnames)

        for fov in fovs:
            # create the dummy data for each fov
            fov_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the preprocessed hdf5
            fov_matrix.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'), key=fov, mode='a')

            # now take 10 random rows from each fov_matrix and add that to pixel_mat_subsetted
            pixel_mat_subsetted = pd.concat([pixel_mat_subsetted, fov_matrix.sample(frac=0.1)])

        pixel_mat_subsetted.to_csv(os.path.join(temp_dir, 'pixel_mat_subsetted.csv'))

        # subset specific markers for clustering
        chan_select = ['Marker1', 'Marker2']

        # run "clustering" using mocked function
        som_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the clustered file has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_clustered.hdf5'))

        # read in this way so we get info about the keys (fovs)
        flowsom_cluster = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_clustered.hdf5'), 'r')

        # make sure the fovs match
        misc_utils.verify_same_elements(
            flowsom_fovs=[fov.replace('/', '') for fov in flowsom_cluster.keys()],
            provided_fovs=fovs)

        # for each fov's data, assert that we didn't assign any cluster 100 or above
        for fov in fovs:
            flowsom_cluster_fov = flowsom_cluster.get(fov)
            cluster_ids = flowsom_cluster_fov['Cluster']

            assert np.all(cluster_ids < 100)
