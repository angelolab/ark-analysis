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
                           subset_dir='pixel_mat_subsetted', norm_vals_name='norm_vals.feather',
                           weights_name='pixel_weights.feather', xdim=10, ydim=10,
                           lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    # define the matrix we'll be training on
    pixel_mat_sub = pd.DataFrame(columns=channels)

    for fov in fovs:
        # read the specific fov from the subsetted HDF5
        fov_mat_sub = feather.read_dataframe(os.path.join(base_dir, subset_dir, fov + '.feather'))

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


def mocked_cluster_pixels(fovs, channels, base_dir, pre_dir='pixel_mat_preprocessed',
                          norm_vals_name='norm_vals.feather', weights_name='pixel_weights.feather',
                          cluster_dir='pixel_mat_clustered',
                          pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv'):
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

        # now assign the calculated cluster_ids as the pixel cluster assignment
        fov_mat_pre['pixel_som_cluster'] = cluster_ids

        # write clustered data to feather
        feather.write_dataframe(fov_mat_pre, os.path.join(base_dir,
                                                          cluster_dir,
                                                          fov + '.feather'))


def mocked_pixel_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                                   cluster_dir='pixel_mat_clustered',
                                   pc_chan_avg_som_cluster_name='pixel_chan_avg_som_cluster.csv',
                                   pc_chan_avg_meta_cluster_name='pixel_chan_avg_meta_cluster.csv',
                                   clust_to_meta_name='pixel_clust_to_meta.feather',
                                   consensus_dir='pixel_mat_consensus', seed=42):
    # read the cluster average
    cluster_avg = pd.read_csv(os.path.join(base_dir, pc_chan_avg_som_cluster_name))

    # dummy scaling using cap
    cluster_avg_scale = cluster_avg[channels] * (cap - 1) / cap

    # get the mean weight for each channel column
    cluster_means = cluster_avg_scale.mean(axis=1)

    # multiply by 100 and mod by 20 to get dummy cluster ids for consensus clustering
    cluster_ids = cluster_means * 100
    cluster_ids = cluster_ids.astype(int) % 20

    # map SOM cluster ids to hierarchical cluster ids
    hClust_to_clust = cluster_avg.drop(columns=channels)
    hClust_to_clust['pixel_meta_cluster'] = cluster_ids

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


def mocked_train_cell_som(fovs, base_dir, pixel_consensus_dir, cell_table_name,
                          cluster_counts_name='cluster_counts.feather',
                          cluster_counts_norm_name='cluster_counts_norm.feather',
                          pixel_cluster_col='pixel_meta_cluster',
                          pc_chan_avg_name='pc_chan_avg.feather',
                          weights_name='cell_weights.feather',
                          weighted_cell_channel_avg_name='weighted_cell_channel_avg.csv',
                          xdim=10, ydim=10, lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    # read in the cluster counts
    cluster_counts_data = feather.read_dataframe(os.path.join(base_dir, cluster_counts_norm_name))

    # get the cluster columns
    cluster_cols = cluster_counts_data.filter(regex=("cluster_|hCluster_cap_")).columns.values

    # subset cluster_counts by the cluster columns
    cluster_counts_sub = cluster_counts_data[cluster_cols]

    # FlowSOM flattens the weights dimensions, ex. 10x10x10 becomes 100x10
    weights = np.random.rand(100, len(cluster_cols))

    # get the 99.9% normalized values and divide weights by that
    weights = weights / np.quantile(weights, 0.999, axis=0)

    # take 100 random rows from cluster_counts_sub
    # element-wise multiply weights by that and num_passes
    multiply_factor = cluster_counts_sub.sample(n=100, replace=True).values
    weights = weights * multiply_factor * num_passes

    # write weights to feather, the result in R will be more like a DataFrame
    weights = pd.DataFrame(weights, columns=cluster_cols)
    feather.write_dataframe(weights, os.path.join(base_dir, weights_name))


def mocked_cluster_cells(base_dir, cluster_counts_norm_name='cluster_counts_norm.feather',
                         weights_name='cell_weights.feather',
                         cell_cluster_name='cell_mat_clustered.feather',
                         pixel_cluster_col_prefix='pixel_meta_cluster',
                         cell_som_cluster_avgs_name='cell_som_cluster_avgs.feather'):
    # read in the cluster counts data
    cluster_counts = feather.read_dataframe(os.path.join(base_dir, cluster_counts_norm_name))

    # read in the weights matrix
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))

    # get the mean weight for each channel column
    sub_means = weights.mean(axis=1)

    # multiply by 100 and truncate to int to get an actual cluster id
    cluster_ids = sub_means * 100
    cluster_ids = cluster_ids.astype(int)

    # assign as cell cluster assignment
    cluster_counts['cell_som_cluster'] = cluster_ids

    # write clustered data to feather
    feather.write_dataframe(cluster_counts, os.path.join(base_dir, cell_cluster_name))


def mocked_cell_consensus_cluster(fovs, channels, base_dir, pixel_cluster_col, max_k=20, cap=3,
                                  cell_cluster_name='cell_mat_clustered.feather',
                                  cell_som_cluster_avgs_name='cell_som_cluster_avgs.csv',
                                  cell_meta_cluster_avgs_name='cell_meta_cluster_avgs.csv',
                                  cell_cluster_col='cell_meta_cluster',
                                  weighted_cell_channel_avg_name='weighted_cell_channel_avg.csv',
                                  cell_cluster_channel_avg_name='cell_cluster_channel_avg.csv',
                                  clust_to_meta_name='cell_clust_to_meta.feather',
                                  cell_consensus_name='cell_mat_consensus.feather', seed=42):
    # read in the cluster averages
    cluster_avg = pd.read_csv(os.path.join(base_dir, cell_som_cluster_avgs_name))

    # dummy scaling using cap
    cluster_avg_scale = cluster_avg.filter(
        regex=("pixel_som_cluster_|pixel_meta_cluster_")
    ) * (cap - 1) / cap

    # get the mean weight for each channel column
    cluster_means = cluster_avg_scale.mean(axis=1)

    # multiply by 100 and mod by 2 to get dummy cluster ids for consensus clustering
    cluster_ids = cluster_means * 100
    cluster_ids = cluster_ids.astype(int) % 2

    # read in the original cell data
    cell_data = feather.read_dataframe(os.path.join(base_dir, cell_cluster_name))

    # add hCluster_cap labels
    cell_data['cell_meta_cluster'] = np.repeat(cluster_ids.values, 10)

    # write cell_data
    feather.write_dataframe(cell_data, os.path.join(base_dir, cell_consensus_name))


def test_normalize_rows():
    # define a list of channels and a subset of channels
    chans = ['chan0', 'chan1', 'chan2']
    chan_sub = chans[:2]

    # create a dummy pixel matrix
    fov_pixel_matrix = pd.DataFrame(
        np.repeat(np.array([[2, 2, 4]]), repeats=1000, axis=0),
        columns=chans
    )

    # add dummy metadata
    fov_pixel_matrix['fov'] = 'fov0'
    fov_pixel_matrix['row_index'] = -1
    fov_pixel_matrix['column_index'] = -1
    fov_pixel_matrix['segmentation_label'] = -1

    # define the meta cols for ease of use
    meta_cols = ['fov', 'row_index', 'column_index', 'segmentation_label']

    # TEST 1: normalize the matrix and keep the segmentation_label column
    # NOTE: this test errors out if 'segmentation_label' is not included in fov_pixel_matrix_sub
    fov_pixel_matrix_sub = som_utils.normalize_rows(fov_pixel_matrix, chan_sub)

    # assert the same channels we subsetted on are found in fov_pixel_matrix_sub
    misc_utils.verify_same_elements(
        provided_chans=chan_sub,
        fov_pixel_chans=fov_pixel_matrix_sub.drop(columns=meta_cols).columns.values
    )

    # assert all the rows sum to 0.5, 0.5
    # this also checks that all the zero-sum rows have been removed
    assert np.all(fov_pixel_matrix_sub.drop(columns=meta_cols).values == [0.5, 0.5])

    # TEST 2: normalize the matrix and drop the segmentation_label column
    meta_cols.remove('segmentation_label')

    fov_pixel_matrix_sub = som_utils.normalize_rows(
        fov_pixel_matrix, chan_sub, include_seg_label=False
    )

    # assert the same channels we subsetted on are found in fov_pixel_matrix_sub
    misc_utils.verify_same_elements(
        provided_chans=chan_sub,
        fov_pixel_chans=fov_pixel_matrix_sub.drop(columns=meta_cols).columns.values
    )

    # assert all the rows sum to 0.5, 0.5
    # this also checks that all the zero-sum rows have been removed
    assert np.all(fov_pixel_matrix_sub.drop(columns=meta_cols).values == [0.5, 0.5])


def test_compute_pixel_cluster_channel_avg():
    # define list of fovs and channels
    fovs = ['fov0', 'fov1', 'fov2']
    chans = ['chan0', 'chan1', 'chan2']

    # do not need to test for cluster_dir existence, that happens in consensus_cluster
    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel cluster col passed
        with pytest.raises(ValueError):
            som_utils.compute_pixel_cluster_channel_avg(
                fovs, chans, 'base_dir', 'bad_cluster_col', temp_dir, False
            )

        # create a dummy pixel and meta clustered matrix
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_consensus'))

        # write dummy clustered data for each fov
        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_cluster_matrix = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=1000, axis=0),
                columns=chans
            )

            # assign dummy SOM cluster labels
            fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(100), repeats=10)

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_clustered',
                                                                     fov + '.feather'))

            # assign dummy meta cluster labels
            fov_cluster_matrix['pixel_meta_cluster'] = np.repeat(np.arange(10), repeats=100)

            # write the dummy data to pixel_mat_consensus
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_consensus',
                                                                     fov + '.feather'))

        for cluster_col in ['pixel_som_cluster', 'pixel_meta_cluster']:
            # define the final result we should get
            if cluster_col == 'pixel_som_cluster':
                num_repeats = 100
            else:
                num_repeats = 10

            result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)

            for keep_count in [False, True]:
                # compute pixel cluster average matrix
                cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
                    fovs, chans, temp_dir, cluster_col,
                    'pixel_mat_consensus', keep_count=keep_count
                )

                # verify the provided channels and the channels in cluster_avg are exactly the same
                misc_utils.verify_same_elements(
                    cluster_avg_chans=cluster_avg[chans].columns.values,
                    provided_chans=chans
                )

                # define the columns to check in cluster_avg, count may also be included
                cluster_avg_cols = chans[:]

                # if keep_count is true then add the counts
                if keep_count:
                    if cluster_col == 'pixel_som_cluster':
                        counts = 30
                    else:
                        counts = 300

                    count_col = np.expand_dims(np.repeat(counts, repeats=result.shape[0]), axis=1)
                    result = np.append(result, count_col, 1)

                    cluster_avg_cols.append('count')

                # assert all elements of cluster_avg and the actual result are equal
                assert np.array_equal(result, np.round(cluster_avg[cluster_avg_cols].values, 1))

        # compute pixel cluster average matrix
        cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, 'pixel_som_cluster', 'pixel_mat_clustered'
        )

        # verify the provided channels and the channels in cluster_avg are exactly the same
        misc_utils.verify_same_elements(
            cluster_avg_chans=cluster_avg[chans].columns.values,
            provided_chans=chans
        )

        # assert all rows equal [0.1, 0.2, 0.3], round due to minor float arithmetic inaccuracies
        result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=100, axis=0)
        assert np.array_equal(result, np.round(cluster_avg[chans].values, 1))

        # compute meta cluster average matrix
        cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, 'pixel_meta_cluster', 'pixel_mat_consensus'
        )

        # verify the provided channels and the channels in cluster_avg are exactly the same
        misc_utils.verify_same_elements(
            cluster_avg_chans=cluster_avg[chans].columns.values,
            provided_chans=chans
        )

        # assert all rows equal [0.1, 0.2, 0.3], round due to minor float arithmetic inaccuracies
        result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=10, axis=0)
        assert np.array_equal(result, np.round(cluster_avg[chans].values, 1))


def test_compute_cell_cluster_count_avg():
    # define the cluster columns
    pixel_som_clusters = ['pixel_som_cluster_%d' % i for i in np.arange(3)]
    pixel_meta_clusters = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel_cluster_col_prefix specified
        with pytest.raises(ValueError):
            som_utils.compute_cell_cluster_count_avg(
                'clustered_path', 'bad_cluster_col_prefix', 'cell_cluster_col', False
            )

        # error check: bad cell_cluster_col specified
        with pytest.raises(ValueError):
            som_utils.compute_cell_cluster_count_avg(
                'clustered_path', 'pixel_meta_cluster', 'bad_cluster_col', False
            )

        cluster_col_arr = [pixel_som_clusters, pixel_meta_clusters]

        # test for both pixel SOM and meta clusters
        for i in range(len(cluster_col_arr)):
            cluster_prefix = 'pixel_som_cluster' if i == 0 else 'pixel_meta_cluster'

            # create a dummy cluster_data file
            cluster_data = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.1, 0.1]]), repeats=1000, axis=0),
                columns=cluster_col_arr[i]
            )

            # add metadata, for cell cluster averaging the values don't matter
            cluster_data['fov'] = 'fov'
            cluster_data['row_index'] = -1
            cluster_data['column_index'] = -1
            cluster_data['segmentation_label'] = -1

            # assign cell cluster labels
            cluster_data['cell_som_cluster'] = np.repeat(np.arange(10), 100)
            cluster_data['cell_meta_cluster'] = np.repeat(np.arange(2), 500)

            # write cluster data
            clustered_path = os.path.join(temp_dir, 'cell_mat_clustered.feather')
            feather.write_dataframe(cluster_data, clustered_path)

            # test for both keep_count settings
            for keep_count in [False, True]:
                # TEST 1: paveraged over cell SOM clusters
                # drop a certain set of columns when checking count avg values
                drop_cols = ['cell_som_cluster']
                if keep_count:
                    drop_cols.append('count')

                cell_cluster_avg = som_utils.compute_cell_cluster_count_avg(
                    clustered_path, cluster_prefix, 'cell_som_cluster', keep_count=keep_count
                )

                # assert we have results for all 10 labels
                assert cell_cluster_avg.shape[0] == 10

                # assert the values are 0.1 across the board
                cell_cluster_avg_sub = cell_cluster_avg.drop(columns=drop_cols)

                # division causes tiny errors so round to 1 decimal place
                cell_cluster_avg_sub = cell_cluster_avg_sub.round(decimals=1)

                assert np.all(cell_cluster_avg_sub == 0.1)

                # assert that the counts are valid if keep_count set to True
                if keep_count:
                    assert np.all(cell_cluster_avg['count'].values == 100)

                # TEST 2: averaged over cell meta clusters
                # drop a certain set of columns when checking count avg values
                drop_cols = ['cell_meta_cluster']
                if keep_count:
                    drop_cols.append('count')

                cell_cluster_avg = som_utils.compute_cell_cluster_count_avg(
                    clustered_path, cluster_prefix, 'cell_meta_cluster', keep_count=keep_count
                )

                # assert we have results for all 2 labels
                assert cell_cluster_avg.shape[0] == 2

                # assert the values are 0.1 across the board
                cell_cluster_avg_sub = cell_cluster_avg.drop(columns=drop_cols)

                # division causes tiny errors so round to 1 decimal place
                cell_cluster_avg_sub = cell_cluster_avg_sub.round(decimals=1)

                assert np.all(cell_cluster_avg_sub == 0.1)

                # assert that the counts are valid if keep_count set to True
                if keep_count:
                    assert np.all(cell_cluster_avg['count'].values == 500)


def test_compute_cell_cluster_channel_avg():
    fovs = ['fov1', 'fov2']
    chans = ['chan1', 'chan2', 'chan3']

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: no channel average file provided
        with pytest.raises(FileNotFoundError):
            som_utils.compute_cell_cluster_channel_avg(
                fovs, chans, temp_dir, 'bad_cell_table', 'cell_consensus', 'bad_cluster_col'
            )

        # create an example weighted cell table
        weighted_cell_table = pd.DataFrame(
            np.random.rand(10, 3),
            columns=chans
        )

        # assign dummy fovs
        weighted_cell_table.loc[0:4, 'fov'] = 'fov1'
        weighted_cell_table.loc[5:9, 'fov'] = 'fov2'

        # assign dummy segmentation labels, 5 cells for each
        weighted_cell_table.loc[0:4, 'segmentation_label'] = np.arange(5)
        weighted_cell_table.loc[5:9, 'segmentation_label'] = np.arange(5)

        # assign dummy cell sizes, these won't really matter for this test
        weighted_cell_table['cell_size'] = 5

        # write the data to csv
        weighted_cell_table.to_csv(os.path.join(temp_dir, 'weighted_cell_channel_avg.csv'))

        # error check: bad cell_cluster_col provided
        with pytest.raises(ValueError):
            som_utils.compute_cell_cluster_channel_avg(
                fovs, chans, temp_dir, 'weighted_cell_channel_avg.csv',
                'cell_consensus', cell_cluster_col='bad_cluster_col'
            )

        # create a dummy cell consensus data file
        # the actual column prefix won't matter for this test
        consensus_data = pd.DataFrame(
            np.random.randint(0, 100, (10, 3)),
            columns=['pixel_meta_cluster_%d' % i for i in np.arange(3)]
        )

        # assign dummy cell cluster labels
        consensus_data['cell_som_cluster'] = np.repeat(np.arange(5), 2)

        # assign dummy consensus cluster labels
        consensus_data['cell_meta_cluster'] = np.repeat(np.arange(2), 5)

        # write consensus data
        consensus_path = os.path.join(temp_dir, 'cell_mat_consensus.feather')
        feather.write_dataframe(consensus_data, consensus_path)

        # test averages for cell SOM clusters
        cell_channel_avg = som_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel_avg.csv',
            'cell_mat_consensus.feather', cell_cluster_col='cell_som_cluster'
        )

        # assert the same SOM clusters were assigned
        assert np.all(cell_channel_avg['cell_som_cluster'].values == np.arange(5))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_som_cluster').shape == (5, 3)

        # test averages for cell meta clusters
        cell_channel_avg = som_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel_avg.csv',
            'cell_mat_consensus.feather', cell_cluster_col='cell_meta_cluster'
        )

        # assert the same meta clusters were assigned
        assert np.all(cell_channel_avg['cell_meta_cluster'].values == np.arange(2))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_meta_cluster').shape == (2, 3)


def test_compute_p2c_weighted_channel_avg():
    fovs = ['fov1', 'fov2']
    chans = ['chan1', 'chan2', 'chan3']

    # create an example cell table
    cell_table = pd.DataFrame(np.random.rand(10, 3), columns=chans)

    # assign dummy fovs
    cell_table.loc[0:4, 'fov'] = 'fov1'
    cell_table.loc[5:9, 'fov'] = 'fov2'

    # assign dummy segmentation labels, 5 cells for each
    cell_table.loc[0:4, 'label'] = np.arange(5)
    cell_table.loc[5:9, 'label'] = np.arange(5)

    # assign dummy cell sizes, these won't really matter for this test
    cell_table['cell_size'] = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        # write cell table
        cell_table_path = os.path.join(temp_dir, 'cell_table_size_normalized.csv')
        cell_table.to_csv(cell_table_path, index=False)

        # define a consensus clustered directory
        pixel_consensus_path = os.path.join(temp_dir, 'pixel_consensus_path')
        os.mkdir(pixel_consensus_path)

        # create dummy data for each fov
        for fov in ['fov1', 'fov2']:
            # assume each label has 10 pixels, create dummy data for each of them
            fov_table = pd.DataFrame(
                np.tile(np.array([0.1, 0.2, 0.4]), 50).reshape(50, 3),
                columns=chans
            )

            # assign the fovs and labels
            fov_table['fov'] = fov
            fov_table['segmentation_label'] = np.repeat(np.arange(5), 10)

            # assign dummy pixel/meta labels
            # pixel: 0-1 for fov1 and 1-2 for fov2
            # meta: 0-1 for both fov1 and fov2
            # note: defining them this way greatly simplifies testing
            if fov == 'fov1':
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(2), 25)
            else:
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(1, 3), 25)

            fov_table['pixel_meta_cluster'] = np.repeat(np.arange(2), 25)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_consensus_path,
                                                            fov + '.feather'))

        # iterate over both cluster col vals
        for cluster_col in ['pixel_som_cluster', 'pixel_meta_cluster']:
            # count number of clusters for each cell
            cell_counts, _ = som_utils.create_c2pc_data(
                fovs, pixel_consensus_path, cell_table_path, pixel_cluster_col=cluster_col
            )

            # compute average cluster expression for each pixel som cluster
            cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
                fovs, chans, temp_dir, cluster_col, 'pixel_consensus_path'
            )

            # error check: invalid fovs provided
            with pytest.raises(ValueError):
                som_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, cell_counts, fovs=['fov2', 'fov3']
                )

            # error check: invalid pixel_cluster_col provided
            with pytest.raises(ValueError):
                som_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, cell_counts, pixel_cluster_col='bad_cluster_col'
                )

            # test for all and some fovs
            for fov_list in [None, fovs[:1]]:
                # test with som cluster counts and all fovs
                channel_avg = som_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, cell_counts, fovs=fov_list, pixel_cluster_col=cluster_col
                )

                # subset over just the marker values
                channel_avg_markers = channel_avg[chans].values

                # define the actual values, num rows will be different depending on fov_list
                if fov_list is None:
                    num_repeats = 10
                else:
                    num_repeats = 5

                actual_markers = np.tile(
                    np.array([1, 2, 4]), num_repeats
                ).reshape(num_repeats, 3)

                # assert the values are close enough
                assert np.allclose(channel_avg_markers, actual_markers)


def test_create_c2pc_data():
    fovs = ['fov1', 'fov2']
    chans = ['chan1', 'chan2', 'chan3']

    # create an example cell table
    cell_table = pd.DataFrame(np.random.rand(10, 3), columns=chans)

    # assign dummy fovs
    cell_table.loc[0:4, 'fov'] = 'fov1'
    cell_table.loc[5:9, 'fov'] = 'fov2'

    # assign dummy segmentation labels, 5 cells for each
    cell_table.loc[0:4, 'label'] = np.arange(5)
    cell_table.loc[5:9, 'label'] = np.arange(5)

    # assign dummy cell sizes
    cell_table['cell_size'] = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel_cluster_col provided
        with pytest.raises(ValueError):
            som_utils.create_c2pc_data(
                fovs, 'consensus', 'cell_table', pixel_cluster_col='bad_col'
            )

        # write cell table
        cell_table_path = os.path.join(temp_dir, 'cell_table_size_normalized.csv')
        cell_table.to_csv(cell_table_path, index=False)

        # define a consensus clustered directory
        pixel_consensus_path = os.path.join(temp_dir, 'pixel_consensus_path')
        os.mkdir(pixel_consensus_path)

        # create dummy data for each fov
        for fov in ['fov1', 'fov2']:
            # assume each label has 10 pixels, create dummy data for each of them
            fov_table = pd.DataFrame(np.random.rand(50, 3), columns=chans)

            # assign the fovs and labels
            fov_table['fov'] = fov
            fov_table['segmentation_label'] = np.repeat(np.arange(5), 10)

            # assign dummy pixel/meta labels
            # pixel: 0-1 for fov1 and 1-2 for fov2
            # meta: 0-1 for both fov1 and fov2
            if fov == 'fov1':
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(2), 25)
            else:
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(1, 3), 25)

            fov_table['pixel_meta_cluster'] = np.repeat(np.arange(2), 25)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_consensus_path,
                                                            fov + '.feather'))

        # test counts on the pixel cluster column
        cluster_counts, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_consensus_path, cell_table_path, pixel_cluster_col='pixel_som_cluster'
        )

        # assert we actually created the cluster_cols
        cluster_cols = ['pixel_som_cluster_' + str(cluster_num) for cluster_num in np.arange(3)]
        misc_utils.verify_in_list(
            cluster_id_cols=cluster_cols,
            cluster_counts_columns=cluster_counts.columns.values
        )

        # assert the values created
        correct_val = [[10, 0, 0],
                       [10, 0, 0],
                       [5, 5, 0],
                       [0, 10, 0],
                       [0, 10, 0],
                       [0, 10, 0],
                       [0, 10, 0],
                       [0, 5, 5],
                       [0, 0, 10],
                       [0, 0, 10]]

        assert np.all(
            np.equal(np.array(correct_val), cluster_counts[cluster_cols].values)
        )
        assert np.all(
            np.equal(np.array(correct_val) / 5, cluster_counts_norm[cluster_cols].values)
        )

        # test counts on the consensus cluster column
        cluster_counts, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_consensus_path, cell_table_path, pixel_cluster_col='pixel_meta_cluster'
        )

        # assert we actually created the hCluster_cap cols
        hCluster_cols = ['pixel_meta_cluster_' + str(cluster_num) for cluster_num in np.arange(2)]
        misc_utils.verify_in_list(
            hCluster_id_cols=hCluster_cols,
            hCluster_counts_columns=cluster_counts.columns.values
        )

        # assert the values created
        correct_val = [[10, 0],
                       [10, 0],
                       [5, 5],
                       [0, 10],
                       [0, 10],
                       [10, 0],
                       [10, 0],
                       [5, 5],
                       [0, 10],
                       [0, 10]]

        assert np.all(
            np.equal(np.array(correct_val), cluster_counts[hCluster_cols].values)
        )
        assert np.all(
            np.equal(np.array(correct_val) / 5, cluster_counts_norm[hCluster_cols].values)
        )


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

        # TEST 1: run fov preprocessing for one fov with seg_labels
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

        # TEST 2: run fov preprocessing for one fov without seg_labels
        sample_pixel_mat, sample_pixel_mat_subset = som_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=None
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-3],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-3],
                                        provided_chans=chans)

        # assert no rows sum to 0
        assert np.all(sample_pixel_mat.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        assert sample_pixel_mat.shape[0] * 0.1 == sample_pixel_mat_subset.shape[0]


# TODO: leaving out MIBItiff testing until someone needs it
def test_create_pixel_matrix():
    # tests for all fovs and some fovs, and all chans and some chans
    fov_lists = [['fov0', 'fov1', 'fov2'], ['fov0', 'fov1'], ['fov0']]
    chan_lists = [['chans0', 'chans1', 'chans2'], ['chans0', 'chans1'], ['chans0']]

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
                                          channels=['chan1'],
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          seg_dir=seg_dir,
                                          subset_proportion=1.1)

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          channels=['chan1'],
                                          base_dir='bad_base_dir',
                                          tiff_dir=tiff_dir,
                                          seg_dir=seg_dir)

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          channels=['chan1'],
                                          base_dir=temp_dir,
                                          tiff_dir='bad_tiff_dir',
                                          seg_dir=seg_dir)

        # test all fovs and channels as well as subsets of fovs and/or channels
        for fovs in fov_lists:
            for chans in chan_lists:
                for sub_dir in [None, 'TIFs']:
                    for seg_dir in [None, temp_dir]:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # create a directory to store the image data
                            tiff_dir = os.path.join(temp_dir, 'TIFs')
                            os.mkdir(tiff_dir)

                            # create a dummy seg_dir
                            seg_dir = os.path.join(temp_dir, 'segmentation')
                            os.mkdir(seg_dir)

                            # create sample image data
                            test_utils.create_paired_xarray_fovs(
                                base_dir=tiff_dir, fov_names=fovs,
                                channel_names=chan_lists[0], sub_dir=sub_dir, img_shape=(10, 10)
                            )

                            # create sample segmentation data
                            for fov in fovs:
                                rand_img = np.random.randint(0, 16, size=(10, 10))
                                file_name = fov + "_feature_0.tif"
                                io.imsave(os.path.join(seg_dir, file_name), rand_img)

                            # pass invalid fov names
                            with pytest.raises(FileNotFoundError):
                                som_utils.create_pixel_matrix(fovs=['fov1', 'fov2', 'fov3'],
                                                              channels=chans,
                                                              base_dir=temp_dir,
                                                              tiff_dir=tiff_dir,
                                                              img_sub_folder=sub_dir,
                                                              seg_dir=seg_dir)

                            # create the pixel matrices
                            som_utils.create_pixel_matrix(fovs=fovs,
                                                          channels=chans,
                                                          base_dir=temp_dir,
                                                          tiff_dir=tiff_dir,
                                                          img_sub_folder=sub_dir,
                                                          seg_dir=seg_dir)

                            # check that we actually created a preprocessed directory
                            assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

                            # check that we actually created a subsetted directory
                            assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

                            for fov in fovs:
                                fov_pre_path = os.path.join(
                                    temp_dir, 'pixel_mat_preprocessed', fov + '.feather'
                                )
                                fov_sub_path = os.path.join(
                                    temp_dir, 'pixel_mat_subsetted', fov + '.feather'
                                )

                                # assert we actually created a .feather preprocessed file
                                # for each fov
                                assert os.path.exists(fov_pre_path)

                                # assert that we actually created a .feather subsetted file
                                # for each fov
                                assert os.path.exists(fov_sub_path)

                                # get the data for the specific fov
                                flowsom_pre_fov = feather.read_dataframe(fov_pre_path)
                                flowsom_sub_fov = feather.read_dataframe(fov_sub_path)

                                # assert the channel names are the same
                                chan_index_stop = -3 if seg_dir is None else -4
                                misc_utils.verify_same_elements(
                                    flowsom_chans=flowsom_pre_fov.columns.values[:chan_index_stop],
                                    provided_chans=chans
                                )

                                # assert no rows sum to 0
                                assert np.all(flowsom_pre_fov.loc[:, chans].sum(
                                    axis=1
                                ).values != 0)

                                # assert the subsetted DataFrame size is 0.1
                                # of the preprocessed DataFrame
                                assert flowsom_pre_fov.shape[0] * 0.1 == flowsom_sub_fov.shape[0]


def test_train_pixel_som(mocker):
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.train_pixel_som(fovs=['fov0'], channels=['Marker1'],
                                      base_dir=temp_dir, subset_dir='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

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
            som_utils.train_pixel_som(fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir)

        # column mismatch between provided channels and subsetted data
        with pytest.raises(ValueError):
            som_utils.train_pixel_som(fovs=fovs, channels=['Marker3', 'Marker4', 'MarkerBad'],
                                      base_dir=temp_dir)

        # add mocked function to "train" the SOM based on dummy subsetted data
        mocker.patch('ark.phenotyping.som_utils.train_pixel_som', mocked_train_pixel_som)

        # run "training" using mocked function
        som_utils.train_pixel_som(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the weights file has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_weights.feather'))

        # assert that the dimensions of the weights are correct
        weights = feather.read_dataframe(os.path.join(temp_dir, 'pixel_weights.feather'))
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
            som_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, pre_dir='bad_path'
            )

        # create a preprocessed directory for the undefined weights test
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, norm_vals_name='bad_path.feather'
            )

        norm_vals = pd.DataFrame(np.random.rand(1, 2), columns=['Marker1', 'Marker2'])
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, weights_name='bad_path.feather'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

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
            norm_vals = pd.DataFrame(
                np.random.rand(1, 4),
                columns=['Marker2', 'Marker3', 'Marker4', 'Marker5']
            )
            feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

            weights = pd.DataFrame(
                np.random.rand(100, 4), columns=['Marker2', 'Marker3', 'Marker4', 'Marker1']
            )
            feather.write_dataframe(weights, os.path.join(temp_dir, 'pixel_weights.feather'))

            # bad column name passed for norm_vals
            som_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

            # column name ordering mismatch for weights
            som_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

            # not all the provided fovs exist
            som_utils.cluster_pixels(fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir)

        # create a dummy normalized values matrix and write to feather
        norm_vals = pd.DataFrame(np.ones((1, 4)), columns=chan_list)
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'norm_vals.feather'))

        # create a dummy weights matrix and write to feather
        weights = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
        feather.write_dataframe(weights, os.path.join(temp_dir, 'pixel_weights.feather'))

        # make a dummy cluster dir
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_clustered'))

        # add mocked function to "cluster" preprocessed data based on dummy weights
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # run "clustering" using mocked function
        som_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        # assert the clustered directory has been created
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_clustered'))

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_clustered',
                                                                   fov + '.feather'))

            # assert we didn't assign any cluster 100 or above
            cluster_ids = fov_cluster_data['pixel_som_cluster']
            assert np.all(cluster_ids < 100)


def test_pixel_consensus_cluster(mocker):
    # basic error checks: bad path to clustered dir
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.pixel_consensus_cluster(fovs=['fov0'], channels=['chan0'],
                                              base_dir=temp_dir, cluster_dir='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

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
            fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(100), repeats=10)

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_clustered',
                                                                     fov + '.feather'))

        # compute averages by cluster, this happens before call to R
        cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, 'pixel_som_cluster'
        )

        # save the DataFrame
        cluster_avg.to_csv(os.path.join(temp_dir, 'pixel_chan_avg_som_cluster.csv'))

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
                fov_cluster_data['pixel_som_cluster'].values ==
                fov_consensus_data['pixel_som_cluster'].values
            )

            # assert we didn't assign any cluster 20 or above
            consensus_cluster_ids = fov_consensus_data['pixel_meta_cluster']


def test_train_cell_som(mocker):
    # basic error check: bad path to cell table path
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.train_cell_som(fovs=['fov0'], base_dir=temp_dir,
                                     pixel_consensus_dir='consensus_dir',
                                     cell_table_name='cell_table.csv')

    # basic error check: bad path to consensus dir
    with tempfile.TemporaryDirectory() as temp_dir:
        blank_cell_table = pd.DataFrame()
        blank_cell_table.to_csv(os.path.join(temp_dir, 'sample_cell_table.csv'))

        with pytest.raises(FileNotFoundError):
            som_utils.train_cell_som(fovs=['fov0'], base_dir=temp_dir,
                                     pixel_consensus_dir='consensus_dir',
                                     cell_table_name='sample_cell_table.csv')

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markeres and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov1', 'fov2']

        # create an example cell table
        cell_table = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)

        # assign dummy fovs
        cell_table.loc[0:49, 'fov'] = 'fov1'
        cell_table.loc[50:99, 'fov'] = 'fov2'

        # assign dummy segmentation labels, 50 cells for each
        cell_table.loc[0:49, 'label'] = np.arange(50)
        cell_table.loc[50:99, 'label'] = np.arange(50)

        # assign dummy cell sizes
        cell_table['cell_size'] = np.random.randint(low=1, high=1000, size=(100, 1))

        # write cell table
        cell_table_path = os.path.join(temp_dir, 'cell_table_size_normalized.csv')
        cell_table.to_csv(cell_table_path, index=False)

        # define a consensus clustered directory
        pixel_consensus_path = os.path.join(temp_dir, 'pixel_consensus_dir')
        os.mkdir(pixel_consensus_path)

        # create dummy data for each fov
        for fov in fovs:
            # assume each label has 10 pixels, create dummy data for each of them
            fov_table = pd.DataFrame(np.random.rand(1000, 4), columns=chan_list)

            # assign the fovs and labels
            fov_table['fov'] = fov
            fov_table['segmentation_label'] = np.repeat(np.arange(50), 20)

            # assign dummy pixel/meta labels
            # pixel: 0-9 for fov1 and 5-14 for fov2
            # meta: 0-1 for both fov1 and fov2
            if fov == 'fov1':
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(10), 100)
            else:
                fov_table['pixel_som_cluster'] = np.repeat(np.arange(5, 15), 100)

            fov_table['pixel_meta_cluster'] = np.repeat(np.arange(2), 500)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_consensus_path,
                                                            fov + '.feather'))

        # bad cluster_col provided
        with pytest.raises(ValueError):
            som_utils.train_cell_som(
                fovs, temp_dir, 'pixel_consensus_dir', 'cell_table_size_normalized.csv',
                pixel_cluster_col='bad_cluster'
            )

        # TEST 1: computing weights using pixel clusters
        # compute cluster counts
        _, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_consensus_path, cell_table_path, 'pixel_som_cluster'
        )

        # write cluster count
        cluster_counts_norm_path = os.path.join(temp_dir, 'cluster_counts_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_norm_path)

        # add mocked function to "train_cell_som"
        mocker.patch(
            'ark.phenotyping.som_utils.train_cell_som',
            mocked_train_cell_som
        )

        # "train" the cell SOM using mocked function
        som_utils.train_cell_som(
            fovs=fovs, base_dir=temp_dir, pixel_consensus_dir='pixel_consensus_dir',
            cell_table_name='cell_table_size_normalized.csv',
            pixel_cluster_col='pixel_som_cluster'
        )

        # assert cell weights has been created
        assert os.path.exists(os.path.join(temp_dir, 'cell_weights.feather'))

        # read in the cell weights
        cell_weights = feather.read_dataframe(os.path.join(temp_dir, 'cell_weights.feather'))

        # assert we created the columns needed
        misc_utils.verify_same_elements(
            cluster_col_labels=['pixel_som_cluster_' + str(i) for i in range(15)],
            cluster_weights_names=cell_weights.columns.values
        )

        # assert the shape
        assert cell_weights.shape == (100, 15)

        # remove cell weights for next test
        os.remove(os.path.join(temp_dir, 'cell_weights.feather'))

        # TEST 2: computing weights using hierarchical clusters
        _, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_consensus_path, cell_table_path, 'pixel_meta_cluster'
        )

        # write cluster count
        cluster_counts_norm_path = os.path.join(temp_dir, 'cluster_counts_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_norm_path)

        # add mocked function to "train" cell SOM
        mocker.patch(
            'ark.phenotyping.som_utils.train_cell_som',
            mocked_train_cell_som
        )

        # "train" the cell SOM using mocked function
        som_utils.train_cell_som(
            fovs=fovs, base_dir=temp_dir, pixel_consensus_dir='pixel_consensus_dir',
            cell_table_name='cell_table_size_normalized.csv',
            pixel_cluster_col='pixel_meta_cluster'
        )

        # assert cell weights has been created
        assert os.path.exists(os.path.join(temp_dir, 'cell_weights.feather'))

        # read in the cell weights
        cell_weights = feather.read_dataframe(os.path.join(temp_dir, 'cell_weights.feather'))

        # assert we created the columns needed
        misc_utils.verify_same_elements(
            cluster_col_labels=['pixel_meta_cluster_' + str(i) for i in range(2)],
            cluster_weights_names=cell_weights.columns.values
        )

        # assert the shape
        assert cell_weights.shape == (100, 2)


def test_cluster_cells(mocker):
    # basic error check: path to cell counts norm does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_cells(base_dir=temp_dir, cluster_counts_norm_name='bad_path')

    # basic error check: path to cell weights does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy cluster_counts_norm_name file
        cluster_counts_norm = pd.DataFrame()
        cluster_counts_norm.to_csv(os.path.join(temp_dir, 'cluster_counts_norm.feather'))

        with pytest.raises(FileNotFoundError):
            som_utils.cluster_cells(base_dir=temp_dir,
                                    weights_name='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # define the cluster column names
        cluster_cols = ['pixel_som_cluster_' + str(i) for i in range(3)]

        # create a sample cluster counts file
        cluster_counts = pd.DataFrame(np.random.randint(0, 100, (100, 3)),
                                      columns=cluster_cols)

        # add metadata
        cluster_counts['fov'] = -1
        cluster_counts['cell_size'] = -1
        cluster_counts['segmentation_label'] = -1

        # write cluster counts
        cluster_counts_path = os.path.join(temp_dir, 'cluster_counts.feather')
        feather.write_dataframe(cluster_counts, cluster_counts_path)

        # create normalized counts
        cluster_counts_norm = cluster_counts.copy()
        cluster_counts_norm[cluster_cols] = cluster_counts_norm[cluster_cols] / 5

        # write normalized counts
        cluster_counts_norm_path = os.path.join(temp_dir, 'cluster_counts_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_norm_path)

        with pytest.raises(ValueError):
            bad_cluster_cols = cluster_cols[:]
            bad_cluster_cols[2], bad_cluster_cols[1] = bad_cluster_cols[1], bad_cluster_cols[2]

            weights = pd.DataFrame(np.random.rand(100, 3), columns=bad_cluster_cols)
            feather.write_dataframe(weights, os.path.join(temp_dir, 'cell_weights.feather'))

            # column name mismatch for weights
            som_utils.cluster_cells(base_dir=temp_dir)

        # generate a random weights matrix
        weights = pd.DataFrame(np.random.rand(100, 3), columns=cluster_cols)

        # write weights
        cell_weights_path = os.path.join(temp_dir, 'cell_weights.feather')
        feather.write_dataframe(weights, cell_weights_path)

        # bad cluster_col provided
        with pytest.raises(ValueError):
            som_utils.cluster_cells(
                base_dir=temp_dir,
                pixel_cluster_col_prefix='bad_cluster'
            )

        # add mocked function to "cluster" cells
        mocker.patch(
            'ark.phenotyping.som_utils.cluster_cells',
            mocked_cluster_cells
        )

        # "cluster" the cells
        som_utils.cluster_cells(base_dir=temp_dir)

        # assert the clustered feather file has been created
        assert os.path.exists(os.path.join(temp_dir, 'cell_mat_clustered.feather'))

        # assert we didn't assign any cluster 100 or above
        cell_clustered_data = feather.read_dataframe(
            os.path.join(temp_dir, 'cell_mat_clustered.feather')
        )

        cluster_ids = cell_clustered_data['cell_som_cluster']
        assert np.all(cluster_ids < 100)


def test_cell_consensus_cluster(mocker):
    # basic error check: path to cell clustered does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir,
                cell_cluster_name='bad_path', pixel_cluster_col='blah'
            )

    # basic error check: cell cluster avg table not found
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_data = pd.DataFrame()
            feather.write_dataframe(
                cell_cluster_data, os.path.join(temp_dir, 'cell_mat_clustered.feather')
            )

            som_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col='blah'
            )

    # basic error check: weighted channel avg table not found
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_data = pd.DataFrame()
            cell_cluster_avg_data = pd.DataFrame()
            feather.write_dataframe(
                cell_cluster_data, os.path.join(temp_dir, 'cell_mat_clustered.feather')
            )
            cell_cluster_avg_data.to_csv(
                os.path.join(temp_dir, 'cell_som_cluster_avgs.csv')
            )

            som_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col='blah'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # test for both pixel SOM and meta cluster column names
        for cluster_prefix in ['pixel_som_cluster', 'pixel_meta_cluster']:
            # create a dummy cluster_data file
            cluster_data = pd.DataFrame(
                np.random.randint(0, 100, (100, 3)),
                columns=['%s_%d' % (cluster_prefix, i) for i in np.arange(3)]
            )

            # assign dummy cell cluster labels
            cluster_data['cell_som_cluster'] = np.repeat(np.arange(10), 10)

            # write clustered data
            clustered_path = os.path.join(temp_dir, 'cell_mat_clustered.feather')
            feather.write_dataframe(cluster_data, clustered_path)

            # compute average counts of each pixel SOM/meta cluster across all cell SOM clusters
            cluster_avg = som_utils.compute_cell_cluster_count_avg(
                clustered_path, pixel_cluster_col_prefix=cluster_prefix,
                cell_cluster_col='cell_som_cluster'
            )

            # write cluster average
            cluster_avg_path = os.path.join(temp_dir, 'cell_som_cluster_avgs.csv')
            cluster_avg.to_csv(cluster_avg_path)

            # create a dummy weighted channel average table
            weighted_channel_avg = pd.DataFrame()

            # write dummy weighted channel average table
            weighted_channel_path = os.path.join(temp_dir, 'weighted_cell_channel_avg.csv')
            weighted_channel_avg.to_csv(weighted_channel_path)

            # add mocked function to "consensus cluster" cell average data
            mocker.patch(
                'ark.phenotyping.som_utils.cell_consensus_cluster',
                mocked_cell_consensus_cluster
            )

            # "consensus cluster" the cells
            som_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col=cluster_prefix
            )

            # assert the consensus feather file has been created
            assert os.path.exists(os.path.join(temp_dir, 'cell_mat_consensus.feather'))

            # assert we idn't assign any cluster 2 or above
            cell_consensus_data = feather.read_dataframe(
                os.path.join(temp_dir, 'cell_mat_consensus.feather')
            )

            cluster_ids = cell_consensus_data['cell_meta_cluster']
            assert np.all(cluster_ids < 2)
