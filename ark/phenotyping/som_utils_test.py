import os
import pytest
import tempfile

import feather
import numpy as np
import pandas as pd
import skimage.io as io
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.io_utils as io_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def mocked_train_pixel_som(fovs, channels, base_dir,
                           sub_dir='pixel_mat_subsetted', norm_vals_name='norm_vals.feather',
                           weights_name='weights.feather', num_passes=1):
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

    # get the 99.9% normalized values and divide weights by that
    weights = weights / np.quantile(weights, 0.999, axis=0)

    # save 99.9% normalized values
    norm_vals = np.expand_dims(np.quantile(weights, 0.999, axis=0).T, axis=0)
    quantiles = pd.DataFrame(norm_vals, columns=channels)
    feather.write_dataframe(quantiles, os.path.join(base_dir, norm_vals_name))

    # take 100 random rows from pixel_mat_sub, element-wise multiply weights by that and num_passes
    multiply_factor = pixel_mat_sub.sample(n=100).values
    weights = weights * multiply_factor * num_passes

    # write weights to feather, the result in R will be more like a DataFrame
    weights = pd.DataFrame(weights, columns=channels)
    feather.write_dataframe(weights, os.path.join(base_dir, weights_name))


def mocked_cluster_pixels(fovs, base_dir, pre_dir='pixel_mat_preprocessed',
                          norm_vals_name='norm_vals.feather', weights_name='weights.feather',
                          cluster_dir='pixel_mat_clustered'):
    # read in the norm_vals matrix
    norm_vals = feather.read_dataframe(os.path.join(base_dir, norm_vals_name))

    # read in the weights matrix
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))

    for fov in fovs:
        # read the specific fov from the preprocessed feather
        fov_mat_pre = feather.read_dataframe(os.path.join(base_dir, pre_dir, fov + '.feather'))

        # only take the specified channel columns
        fov_mat_channels = fov_mat_pre[weights.columns.values].copy()

        # perform 99.9% normalization
        fov_mat_channels = fov_mat_channels.div(norm_vals, axis=1)

        # get the mean weight for each channel column
        sub_means = weights.mean(axis=1)

        # multiply by 100 and truncate to int to get an actual cluster id
        cluster_ids = sub_means * 100
        cluster_ids = cluster_ids.astype(int)

        # now assign the calculated cluster_ids as the cluster assignment
        fov_mat_pre['cluster'] = cluster_ids

        # write clustered data to feather
        feather.write_dataframe(fov_mat_pre, os.path.join(base_dir,
                                                          cluster_dir,
                                                          fov + '.feather'))


def mocked_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                             cluster_dir='pixel_mat_clustered',
                             cluster_avg_name='pixel_cluster_avg.feather',
                             consensus_dir='pixel_mat_consensus'):
    # read the cluster average
    cluster_avg = feather.read_dataframe(os.path.join(base_dir, cluster_avg_name))

    # dummy scaling using cap
    cluster_avg_scale = cluster_avg[channels] * (cap - 1) / cap

    # get the mean weight for each channel column
    cluster_means = cluster_avg_scale.mean(axis=1)

    # multiply by 100 and mod by 20 to get dummy cluster ids for consensus clustering
    cluster_ids = cluster_means * 100
    cluster_ids = cluster_ids.astype(int) % 20

    # map SOM cluster ids to hierarchical cluster ids
    hClust_to_clust = cluster_avg.drop(columns=channels)
    hClust_to_clust['hCluster_cap'] = cluster_ids

    for fov in fovs:
        # read fov pixel data with clusters
        fov_cluster_matrix = feather.read_dataframe(os.path.join(base_dir,
                                                                 cluster_dir,
                                                                 fov + '.feather'))

        # use mapping to assign hierarchical cluster ids
        fov_cluster_matrix = pd.merge(fov_cluster_matrix, hClust_to_clust)

        # write consensus cluster results to feather
        feather.write_dataframe(fov_cluster_matrix, os.path.join(base_dir,
                                                                 consensus_dir,
                                                                 fov + '.feather'))


def test_compute_pixel_cluster_avg():
    # define list of fovs and channels
    fovs = ['fov0', 'fov1', 'fov2']
    chans = ['chan0', 'chan1', 'chan2']

    # do not need to test for cluster_dir existence, that happens in consensus_cluster
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy clustered matrix
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))

        # write dummy clustered data for each fov
        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_cluster_matrix = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=1000, axis=0),
                columns=chans
            )

            # assign dummy cluster labels
            fov_cluster_matrix['cluster'] = np.repeat(np.arange(100), repeats=10)

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_clustered',
                                                                     fov + '.feather'))

        # compute cluster average matrix
        cluster_avg = som_utils.compute_pixel_cluster_avg(fovs, chans, temp_dir, 'cluster')

        # verify the provided channels and the channels in cluster_avg are exactly the same
        misc_utils.verify_same_elements(
            cluster_avg_chans=cluster_avg[chans].columns.values,
            provided_chans=chans)

        # assert all rows equal [0.1, 0.2, 0.3], round due to minor float arithmetic inaccuracies
        result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=100, axis=0)
        assert np.array_equal(result, np.round(cluster_avg[chans].values, 1))


def test_create_fov_pixel_data():
    # tests for all fovs and some fovs
    fovs = ['fov0', 'fov1']
    chans = ['chan0', 'chan1', 'chan2']

    # create sample data
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=fovs,
                                                  channel_names=chans)

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=fovs,
                                                  compartment_names=['whole_cell'])

    # test for each fov
    for fov in fovs:
        sample_img_data = sample_img_xr.loc[fov, ...].values.astype(np.float32)

        seg_labels = sample_labels.loc[fov, ...].values.reshape(10, 10)

        # run fov preprocessing for one fov
        sample_pixel_mat, sample_pixel_mat_subset = som_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=seg_labels
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-4],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-4],
                                        provided_chans=chans)

        # assert no rows sum to 0
        assert np.all(sample_pixel_mat.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        assert sample_pixel_mat.shape[0] * 0.1 == sample_pixel_mat_subset.shape[0]


# TODO: leaving out MIBItiff testing until someone needs it
def test_create_pixel_matrix():
    # tests for all fovs and some fovs
    fov_lists = [['fov0', 'fov1', 'fov2'], ['fov0', 'fov1'], ['fov0']]
    chans = ['chan0', 'chan1', 'chan2']

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=fov_lists[0],
                                                  compartment_names=['whole_cell'])

    # basic error checking
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy tiff_dir
        tiff_dir = os.path.join(temp_dir, 'TIFs')
        os.mkdir(tiff_dir)

        # create a dummy seg_dir
        seg_dir = os.path.join(temp_dir, 'segmentation')
        os.mkdir(seg_dir)

        # invalid subset proportion specified
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          seg_dir=seg_dir,
                                          subset_proportion=1.1)

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          base_dir='bad_base_dir',
                                          tiff_dir=tiff_dir,
                                          seg_dir=seg_dir)

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          base_dir=temp_dir,
                                          tiff_dir='bad_tiff_dir',
                                          seg_dir=seg_dir)

    # test all fovs and channels as well as subsets of fovs and/or channels
    for fovs in fov_lists:
        with tempfile.TemporaryDirectory() as temp_dir:
            # create a directory to store the image data
            tiff_dir = os.path.join(temp_dir, 'TIFs')
            os.mkdir(tiff_dir)

            # create a dummy seg_dir
            seg_dir = os.path.join(temp_dir, 'segmentation')
            os.mkdir(seg_dir)

            # create sample image data
            test_utils.create_paired_xarray_fovs(
                base_dir=tiff_dir, fov_names=fovs, channel_names=chans, img_shape=(10, 10)
            )

            # create sample segmentation data
            for fov in fovs:
                rand_img = np.random.randint(0, 16, size=(10, 10))
                file_name = fov + "_feature_0.tif"
                io.imsave(os.path.join(seg_dir, file_name), rand_img)

            # pass invalid fov names
            with pytest.raises(FileNotFoundError):
                som_utils.create_pixel_matrix(fovs=['fov1', 'fov2', 'fov3'],
                                              base_dir=temp_dir,
                                              tiff_dir=tiff_dir,
                                              seg_dir=seg_dir)

            # create the pixel matrices
            som_utils.create_pixel_matrix(fovs=fovs,
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          seg_dir=seg_dir)

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
                assert np.all(flowsom_pre_fov.loc[:, chans].sum(axis=1).values != 0)

                # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
                assert flowsom_pre_fov.shape[0] * 0.1 == flowsom_sub_fov.shape[0]


def test_train_pixel_som(mocker):
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
        mocker.patch('ark.phenotyping.som_utils.train_pixel_som', mocked_train_pixel_som)

        # run "training" using mocked function
        som_utils.train_pixel_som(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the weights file has been created
        assert os.path.exists(os.path.join(temp_dir, 'weights.feather'))

        # assert that the dimensions of the weights are correct
        weights = feather.read_dataframe(os.path.join(temp_dir, 'weights.feather'))
        assert weights.shape == (100, 4)

        # assert that the weights columns are the same as chan_list
        misc_utils.verify_same_elements(weights_channels=weights.columns.values,
                                        provided_channels=chan_list)

        # assert that the normalized file has been created
        assert os.path.exists(os.path.join(temp_dir, 'norm_vals.feather'))

        # assert the shape of norm_vals contains 1 row and number of columns = len(chan_list)
        norm_vals = feather.read_dataframe(os.path.join(temp_dir, 'norm_vals.feather'))
        assert norm_vals.shape == (1, 4)

        # assert the the norm_vals columns are the same as chan_list
        misc_utils.verify_same_elements(norm_vals_channels=norm_vals.columns.values,
                                        provided_channels=chan_list)


def test_cluster_pixels(mocker):
    # basic error checks: bad path to preprocessed data, norm vals matrix, and weights matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], base_dir=temp_dir, pre_dir='bad_path')

        # create a preprocessed directory for the undefined weights test
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(fovs=['fov0'], base_dir=temp_dir,
                                     norm_vals_name='bad_path.feather')

        norm_vals = pd.DataFrame(np.random.rand(1, 2), columns=['Marker1', 'Marker2'])
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

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

        with pytest.raises(ValueError):
            norm_vals = pd.DataFrame(np.random.rand(1, 2), columns=['Marker4', 'Marker5'])
            feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

            weights = pd.DataFrame(np.random.rand(100, 2), columns=['Marker4', 'Marker5'])
            feather.write_dataframe(weights, os.path.join(temp_dir, 'weights.feather'))

            # column name mismatch for norm_vals
            som_utils.cluster_pixels(fovs=fovs, base_dir=temp_dir)

            # column name mismatch for weights
            som_utils.cluster_pixels(fovs=fovs, base_dir=temp_dir)

            # not all the provided fovs exist
            som_utils.cluster_pixels(fovs=['fov2', 'fov3'], base_dir=temp_dir)

        # create a dummy normalized values matrix and write to feather
        norm_vals = pd.DataFrame(np.ones((1, 4)), columns=chan_list)
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

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
            cluster_ids = fov_cluster_data['cluster']
            assert np.all(cluster_ids < 100)


def test_pixel_consensus_cluster(mocker):
    # basic error checks: bad path to clustered dir
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.consensus_cluster(fovs=['fov0'], channels=['chan0'],
                                        base_dir=temp_dir, cluster_dir='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'col_index', 'segmentation_label']

        # create a dummy clustered matrix
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))

        # write dummy clustered data for each fov
        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_cluster_matrix = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.2, 0.3, 0.4]]), repeats=1000, axis=0),
                columns=chans
            )

            # add metadata
            fov_cluster_matrix = pd.concat(
                [fov_cluster_matrix, pd.DataFrame(np.random.rand(1000, 4), columns=meta_colnames)],
                axis=1
            )

            # assign dummy cluster labels
            fov_cluster_matrix['cluster'] = np.repeat(np.arange(100), repeats=10)

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_clustered',
                                                                     fov + '.feather'))

        # compute averages by cluster, this happens before call to R
        cluster_avg = som_utils.compute_pixel_cluster_avg(fovs, chans, temp_dir, 'cluster')

        # save the DataFrame
        feather.write_dataframe(cluster_avg,
                                os.path.join(temp_dir, 'pixel_cluster_avg.feather'),
                                compression='uncompressed')

        # make a dummy consensus dir
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_consensus'))

        # add mocked function to "consensus cluster" data averaged by cluster
        mocker.patch(
            'ark.phenotyping.som_utils.pixel_consensus_cluster',
            mocked_pixel_consensus_cluster
        )

        # run "consensus clustering" using mocked function
        som_utils.pixel_consensus_cluster(fovs=fovs, channels=chans, base_dir=temp_dir)

        # assert the final consensus cluster directory has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_consensus'))

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_clustered',
                                                                   fov + '.feather'))

            fov_consensus_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                     'pixel_mat_consensus',
                                                                     fov + '.feather'))

            # assert we didn't modify the cluster column in the consensus clustered results
            assert np.all(
                fov_cluster_data['cluster'].values == fov_consensus_data['cluster'].values
            )

            # assert we didn't assign any cluster 20 or above
            consensus_cluster_ids = fov_consensus_data['hCluster_cap']


def test_visualize_pixel_cluster_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # create a dummy clustered matrix
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))

        # write dummy clustered data for each fov
        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_cluster_matrix = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.2, 0.3, 0.4]]), repeats=1000, axis=0),
                columns=chans
            )

            # assign dummy SOM cluster labels
            fov_cluster_matrix['cluster'] = np.repeat(np.arange(100), repeats=10)

            # assign dummy hierarchical cluster labels
            fov_cluster_matrix['hCluster_cap'] = np.repeat(np.arange(20), repeats=50)

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_clustered',
                                                                     fov + '.feather'))

        # test visualization for cluster: no saving
        som_utils.visualize_pixel_cluster_data(fovs=fovs, channels=chans, base_dir=temp_dir,
                                               cluster_dir='pixel_mat_clustered',
                                               cluster_col='cluster', save_dir=None)
        assert not os.path.exists(os.path.join(temp_dir, "som_cluster_avgs.png"))

        # test visualization for cluster: saving
        som_utils.visualize_pixel_cluster_data(fovs=fovs, channels=chans, base_dir=temp_dir,
                                               cluster_dir='pixel_mat_clustered',
                                               cluster_col='cluster', save_dir=temp_dir,
                                               save_file="som_cluster_avgs.png")
        assert os.path.exists(os.path.join(temp_dir, "som_cluster_avgs.png"))

        # test visualization for hierarchical cluster: no saving
        som_utils.visualize_pixel_cluster_data(fovs=fovs, channels=chans, base_dir=temp_dir,
                                               cluster_dir='pixel_mat_clustered',
                                               cluster_col='hCluster_cap', save_dir=None)
        assert not os.path.exists(os.path.join(temp_dir, "som_hierarchical_avgs.png"))

        # test visualization for hierarchical cluster: saving
        som_utils.visualize_pixel_cluster_data(fovs=fovs, channels=chans, base_dir=temp_dir,
                                               cluster_dir='pixel_mat_clustered',
                                               cluster_col='hCluster_cap', save_dir=temp_dir,
                                               save_file="som_hierarchical_avgs.png")
        assert os.path.exists(os.path.join(temp_dir, "som_hierarchical_avgs.png"))
