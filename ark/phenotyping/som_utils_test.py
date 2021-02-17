import h5py
import os
import pytest
import tempfile

import feather
import numpy as np
import pandas as pd
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.io_utils as io_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def mocked_train_som(fovs, channels, base_dir,
                     sub_dir='pixel_mat_subsetted', weights_name='weights.feather'):
    # define the matrix we'll be training on
    pixel_mat_sub = pd.DataFrame(columns=channels)

    for fov in fovs:
        # read the specific fov from the subsetted HDF5
        fov_mat_sub = feather.read_dataframe(os.path.join(base_dir, sub_dir, fov + '.feather'))

        # only take the channel columns
        fov_mat_sub = fov_mat_sub[channels]

        # append to pixel_mat_sub
        pixel_mat_sub = pd.concat([pixel_mat_sub, fov_mat_sub])

    # FlowSOM flattens the weights dimensions, ex. 10x10x10 becomes 100x10
    weights = np.random.rand(100, len(channels))

    # take 100 random rows from pixel_mat_sub, and element-wise multiply weights by that
    multiply_factor = pixel_mat_sub.sample(n=100).values
    weights = weights * multiply_factor

    # write weights to feather, the result in R will be more like a DataFrame
    weights = pd.DataFrame(weights, columns=channels)
    feather.write_dataframe(weights, os.path.join(base_dir, weights_name))


def mocked_cluster_pixels(fovs, base_dir, pre_dir='pixel_mat_preprocessed',
                          weights_name='weights.feather', cluster_dir='pixel_mat_clustered'):
    # read in the weights matrix
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))

    for fov in fovs:
        # read the specific fov from the preprocessed feather
        fov_mat_pre = feather.read_dataframe(os.path.join(base_dir, pre_dir, fov + '.feather'))

        # only take the specified channel columns
        fov_mat_pre = fov_mat_pre[weights.columns.values]

        # get the mean weight for each channel column
        sub_means = weights.mean(axis=1)

        # multiply by 100 and truncate to int to get an actual cluster id
        cluster_ids = sub_means * 100
        cluster_ids = cluster_ids.astype(int)

        # now assign the calculated cluster_ids as the cluster assignment
        fov_mat_pre['clusters'] = cluster_ids

        # write clustered data to HDF5
        feather.write_dataframe(fov_mat_pre, os.path.join(base_dir,
                                                          cluster_dir,
                                                          fov + '.feather'))


def test_create_pixel_matrix():
    # tests for all fovs and some fovs
    fov_lists = [['fov0', 'fov1'], ['fov0']]
    chans = ['chan0', 'chan1', 'chan2']

    # make sample data
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=fov_lists[0],
                                                  channel_names=chans)

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=fov_lists[0],
                                                  compartment_names=['whole_cell'])

    # basic error checking
    with tempfile.TemporaryDirectory() as temp_dir:
        # pass invalid fov names
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          fovs=['fov1', 'fov2'])

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, 'bad_base_dir')

        # invalid subset percentage specified
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir,
                                          subset_proportion=1.1)

    # test all fovs and channels as well as subsets of fovs and/or channels
    for fovs in fov_lists:
        with tempfile.TemporaryDirectory() as temp_dir:
            som_utils.create_pixel_matrix(sample_img_xr, sample_labels, temp_dir, fovs=fovs)

            # check that we actually created a preprocessed directory
            assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

            # check that we actually created a subsetted directory
            assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

            for fov in fovs:
                fov_pre_path = os.path.join(temp_dir, 'pixel_mat_preprocessed', fov + '.feather')
                fov_sub_path = os.path.join(temp_dir, 'pixel_mat_subsetted', fov + '.feather')

                # assert we actually created a .feather preprocessed file for each fov
                assert os.path.exists(fov_pre_path)

                # assert that we actually created a .feather subsetted file for each fov
                assert os.path.exists(fov_sub_path)

                # get the data for the specific fov
                flowsom_pre_fov = feather.read_dataframe(fov_pre_path)
                flowsom_sub_fov = feather.read_dataframe(fov_sub_path)

                # assert the channel names are the same
                misc_utils.verify_same_elements(flowsom_chans=flowsom_pre_fov.columns.values[:-4],
                                                provided_chans=chans)

                # assert no rows sum to 0
                assert np.all(flowsom_pre_fov.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

                # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
                assert flowsom_pre_fov.shape[0] * 0.1 == flowsom_sub_fov.shape[0]


def test_train_som(mocker):
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.train_som(fovs=['fov0'], channels=['Marker1'],
                                base_dir=temp_dir, sub_dir='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'col_index', 'segmentation_label']

        # make a dummy sub directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_subsetted'))

        for fov in fovs:
            # create the dummy data for each fov
            fov_sub_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the subsetted data dir
            feather.write_dataframe(fov_sub_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_subsetted',
                                                                 fov + '.feather'))

        # not all of the provided fovs exist
        with pytest.raises(ValueError):
            som_utils.train_som(fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir)

        # column mismatch between provided channels and subsetted data
        with pytest.raises(ValueError):
            som_utils.train_som(fovs=fovs, channels=['Marker3', 'Marker4', 'MarkerBad'],
                                base_dir=temp_dir)

        # add mocked function to "train" the SOM based on dummy subsetted data
        mocker.patch('ark.phenotyping.som_utils.train_som', mocked_train_som)

        # run "training" using mocked function
        som_utils.train_som(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the weights file has been created
        assert os.path.exists(os.path.join(temp_dir, 'weights.feather'))

        # assert that the dimensions of the weights are correct
        weights = feather.read_dataframe(os.path.join(temp_dir, 'weights.feather'))
        assert weights.shape == (100, 4)


def test_cluster_pixels(mocker):
    # basic error checks: bad path to preprocessed and weights matrices
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], base_dir=temp_dir, pre_dir='bad_path')

        # create a preprocessed directory for the undefined weights test
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], base_dir=temp_dir,
                                     weights_name='bad_path.feather')

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'col_index', 'segmentation_label']

        # make a dummy pre dir
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_pre_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the preprocessed data dir
            feather.write_dataframe(fov_pre_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_preprocessed',
                                                                 fov + '.feather'))

        # not all of the provided fovs exist
        with pytest.raises(ValueError):
            weights = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
            feather.write_dataframe(weights, os.path.join(temp_dir, 'weights.feather'))

            som_utils.cluster_pixels(fovs=['fov2', 'fov3'], base_dir=temp_dir)

        # column name mismatch between weights channels and pixel data channels
        with pytest.raises(ValueError):
            weights = pd.DataFrame(np.random.rand(100, 2), columns=['Marker4', 'Marker5'])
            feather.write_dataframe(weights, os.path.join(temp_dir, 'weights.feather'))

            som_utils.cluster_pixels(fovs=fovs, base_dir=temp_dir)

        # create a dummy weights matrix and write to feather
        weights = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
        feather.write_dataframe(weights, os.path.join(temp_dir, 'weights.feather'))

        # make a dummy cluster dir
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))

        # add mocked function to "cluster" preprocessed data based on dummy weights
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # run "clustering" using mocked function
        som_utils.cluster_pixels(fovs=fovs, base_dir=temp_dir)

        # assert the clustered directory has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_clustered'))

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_clustered',
                                                                   fov + '.feather'))

            # assert we didn't assign any cluster 100 or above
            cluster_ids = fov_cluster_data['clusters']
            assert np.all(cluster_ids < 100)
