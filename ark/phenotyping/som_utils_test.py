import h5py
import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def mocked_train_som(fovs, channels, base_dir,
                     subset_name='pixel_mat_subsetted.hdf5', weights_name='weights.hdf5'):
    # define the matrix we'll be training on
    pixel_mat_sub = pd.DataFrame(columns=channels)

    for fov in fovs:
        # read the specific fov from the subsetted HDF5
        fov_mat_sub = pd.read_hdf(os.path.join(base_dir, subset_name), key=fov)

        # only take the channel columns
        fov_mat_sub = fov_mat_sub[channels]

        # append to pixel_mat_sub
        pixel_mat_sub = pd.concat([pixel_mat_sub, fov_mat_sub])

    # FlowSOM flattens the weights dimensions, ex. 10x10x10 becomes 100x10
    weights = np.random.rand(100, len(channels))

    # take 100 random rows from pixel_mat_sub, and element-wise multiply weights by that
    multiply_factor = pixel_mat_sub.sample(n=100).values
    weights = weights * multiply_factor

    # write weights to HDF5
    with h5py.File(os.path.join(base_dir, weights_name), 'w') as hf:
        hf.create_dataset('weights', data=weights)


def mocked_cluster_pixels(fovs, channels, base_dir,
                          pre_name='pixel_mat_preprocessed.hdf5',
                          weights_name='weights.hdf5',
                          cluster_name='pixel_mat_clustered.hdf5'):
    # read in the weights matrix
    with h5py.File(os.path.join(base_dir, weights_name), 'r') as hf:
        weights = hf['weights'].value

    for fov in fovs:
        # read the specific fov from the preprocessed HDF5
        fov_mat_pre = pd.read_hdf(os.path.join(base_dir, pre_name), key=fov)

        # get the means across the rows using only the channels columns
        sub_means = fov_mat_pre[channels].mean(axis=1)

        # multiply by 100 and truncate to int to get an actual cluster id
        cluster_ids = sub_means * 100
        cluster_ids = cluster_ids.astype(int)

        # now assign the calculated cluster_ids as the cluster assignment
        fov_mat_pre['clusters'] = cluster_ids

        # write clustered data to HDF5
        fov_mat_pre.to_hdf(os.path.join(base_dir, cluster_name), key=fov, mode='a')


def test_create_pixel_matrix():
    fov_lists = [['fov0', 'fov1'], ['fov0']]
    chan_lists = [['chan0', 'chan1'], ['chan0']]

    # must be float64 since that's how Candace's data comes in
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=fov_lists[0],
                                                  channel_names=chan_lists[0],
                                                  dtype='float64')

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=fov_lists[0],
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

        # invalid subset percentage specified
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          subset_percent=1.1)

    # test all fovs and channels as well as subsets of fovs and/or channels
    for fovs, chans in zip(fov_lists, chan_lists):
        with tempfile.TemporaryDirectory() as temp_dir:
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          fovs=fovs, channels=chans)

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

            # check that we actually created a subsetted HDF5
            assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted.hdf5'))

            # read in this way so we get info about the keys (fovs)
            flowsom_sub = pd.HDFStore(os.path.join(temp_dir, 'pixel_mat_subsetted.hdf5'), 'r')

            # assert all fovs and channels are covered
            for fov in flowsom_sub.keys():
                # get the subsetted data for the specific fov
                flowsom_sub_fov = flowsom_sub.get(fov)

                # get the preprocessed data for the fov, for checking counts
                flowsom_pre_fov = flowsom_pre.get(fov)

                # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
                assert flowsom_pre_fov.shape[0] * 0.1 == flowsom_sub_fov.shape[0]

            # close the HDF5 file instances
            flowsom_pre.close()
            flowsom_sub.close()


def test_train_som(mocker):
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.train_som(fovs=['fov0'], channels=['Marker1'],
                                base_dir=temp_dir, subset_name='bad_path.hdf5')

    with tempfile.TemporaryDirectory() as temp_dir:
        # add mocked function to "train" the SOM based on dummy subsetted data
        mocker.patch('ark.phenotyping.som_utils.train_som', mocked_train_som)

        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'col_index', 'segmentation_label']

        for fov in fovs:
            # create the dummy data for each fov
            fov_sub_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the preprocessed hdf5
            fov_sub_matrix.to_hdf(os.path.join(temp_dir, 'pixel_mat_subsetted.hdf5'),
                                  key=fov, mode='a')

        # run "training" using mocked function
        som_utils.train_som(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the weights file has been created
        assert os.path.exists(os.path.join(temp_dir, 'weights.hdf5'))

        # assert that the dimensions of the weights are correct
        with h5py.File(os.path.join(temp_dir, 'weights.hdf5'), 'r') as hf:
            weights = hf['weights'][:]
            assert weights.shape == (100, 4)


def test_cluster_pixels(mocker):
    # basic error checks: bad path to preprocessed and weights matrices
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], channels=['Marker1'],
                                     base_dir=temp_dir, pre_name='bad_path.hdf5')

        # write a random HDF5 for the undefined weights test
        random_df = pd.DataFrame(np.random.rand(10, 2))
        random_df.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'),
                         key='fov0', mode='a')

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], channels=['Marker1'],
                                     base_dir=temp_dir, weights_name='bad_path.hdf5')

    with tempfile.TemporaryDirectory() as temp_dir:
        # add mocked function to "cluster" preprocessed data based on dummy weights
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'col_index', 'segmentation_label']

        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_pre_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the preprocessed HDF5
            fov_pre_matrix.to_hdf(os.path.join(temp_dir, 'pixel_mat_preprocessed.hdf5'),
                                  key=fov, mode='a')

        # create a dummy weights matrix and write to HDF5
        weights = np.random.rand(100, 4)

        with h5py.File(os.path.join(temp_dir, 'weights.hdf5'), 'w') as hf:
            hf.create_dataset('weights', data=weights)

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
            cluster_ids = flowsom_cluster_fov['clusters']

            assert np.all(cluster_ids < 100)
