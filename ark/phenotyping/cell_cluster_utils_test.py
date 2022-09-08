import os
import tempfile

import feather
import numpy as np
import pandas as pd
import pytest
from sklearn.utils import shuffle

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils
import ark.utils.misc_utils as misc_utils

parametrize = pytest.mark.parametrize


def mocked_train_cell_som(fovs, channels, base_dir, pixel_data_dir, cell_table_path,
                          cluster_counts_name='cluster_counts.feather',
                          cluster_counts_norm_name='cluster_counts_norm.feather',
                          pixel_cluster_col='pixel_meta_cluster_rename',
                          pc_chan_avg_name='pc_chan_avg.feather',
                          weights_name='cell_weights.feather',
                          weighted_cell_channel_name='weighted_cell_channel.csv',
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
                         pixel_cluster_col_prefix='pixel_meta_cluster_rename',
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
                                  cell_data_name='cell_mat.feather',
                                  cell_som_cluster_avgs_name='cell_som_cluster_avgs.csv',
                                  cell_meta_cluster_avgs_name='cell_meta_cluster_avgs.csv',
                                  cell_cluster_col='cell_meta_cluster',
                                  weighted_cell_channel_name='weighted_cell_channel.csv',
                                  cell_cluster_channel_avg_name='cell_cluster_channel_avg.csv',
                                  clust_to_meta_name='cell_clust_to_meta.feather', seed=42):
    # read in the cluster averages
    cluster_avg = pd.read_csv(os.path.join(base_dir, cell_som_cluster_avgs_name))

    # dummy scaling using cap
    cluster_avg_scale = cluster_avg.filter(
        regex=("pixel_som_cluster_|pixel_meta_cluster_rename_")
    ) * (cap - 1) / cap

    # get the mean weight for each channel column
    cluster_means = cluster_avg_scale.mean(axis=1)

    # multiply by 100 and mod by 2 to get dummy cluster ids for consensus clustering
    cluster_ids = cluster_means * 100
    cluster_ids = cluster_ids.astype(int) % 2

    # read in the original cell data
    cell_data = feather.read_dataframe(os.path.join(base_dir, cell_data_name))

    # add hCluster_cap labels
    cell_data['cell_meta_cluster'] = np.repeat(cluster_ids.values, 10)

    # write cell_data
    feather.write_dataframe(cell_data, os.path.join(base_dir, cell_data_name))


def test_compute_cell_cluster_count_avg():
    # define the cluster columns
    pixel_som_clusters = ['pixel_som_cluster_%d' % i for i in np.arange(3)]
    pixel_meta_clusters = ['pixel_meta_cluster_rename_%s' % str(i) for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel_cluster_col_prefix specified
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_cluster_count_avg(
                'clustered_path', 'bad_cluster_col_prefix', 'cell_cluster_col', False
            )

        # error check: bad cell_cluster_col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_cluster_count_avg(
                'clustered_path', 'pixel_meta_cluster', 'bad_cluster_col', False
            )

        cluster_col_arr = [pixel_som_clusters, pixel_meta_clusters]

        # test for both pixel SOM and meta clusters
        for i in range(len(cluster_col_arr)):
            cluster_prefix = 'pixel_som_cluster' if i == 0 else 'pixel_meta_cluster_rename'

            # create a dummy cluster_data file
            cluster_data = pd.DataFrame(
                np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=1000, axis=0),
                columns=cluster_col_arr[i]
            )

            # add metadata, for cell cluster averaging the values don't matter
            cluster_data['fov'] = 'fov'
            cluster_data['row_index'] = -1
            cluster_data['column_index'] = -1
            cluster_data['segmentation_label'] = -1

            # assign cell cluster labels
            cluster_data['cell_som_cluster'] = np.repeat(np.arange(10), 100)
            cluster_data['cell_meta_cluster'] = np.repeat(np.arange(5), 200)

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

                cell_cluster_avg = cell_cluster_utils.compute_cell_cluster_count_avg(
                    clustered_path, cluster_prefix, 'cell_som_cluster', keep_count=keep_count
                )

                # assert we have results for all 10 labels
                assert cell_cluster_avg.shape[0] == 10

                # assert the values are [0.1, 0.2, 0.3] across the board
                result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=10, axis=0)
                cell_cluster_avg_sub = cell_cluster_avg.drop(columns=drop_cols)

                # division causes tiny errors so round to 1 decimal place
                cell_cluster_avg_sub = cell_cluster_avg_sub.round(decimals=1)

                assert np.all(result == cell_cluster_avg_sub.values)

                # assert that the counts are valid if keep_count set to True
                if keep_count:
                    assert np.all(cell_cluster_avg['count'].values == 100)

                # TEST 2: averaged over cell meta clusters
                # drop a certain set of columns when checking count avg values
                drop_cols = ['cell_meta_cluster']
                if keep_count:
                    drop_cols.append('count')

                cell_cluster_avg = cell_cluster_utils.compute_cell_cluster_count_avg(
                    clustered_path, cluster_prefix, 'cell_meta_cluster', keep_count=keep_count
                )

                # assert we have results for all 5 labels
                assert cell_cluster_avg.shape[0] == 5

                # assert the values are [0.1, 0.2, 0.3] across the board
                result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=5, axis=0)
                cell_cluster_avg_sub = cell_cluster_avg.drop(columns=drop_cols)

                # division causes tiny errors so round to 1 decimal place
                cell_cluster_avg_sub = cell_cluster_avg_sub.round(decimals=1)

                assert np.all(result == cell_cluster_avg_sub.values)

                # assert that the counts are valid if keep_count set to True
                if keep_count:
                    assert np.all(cell_cluster_avg['count'].values == 200)


def test_compute_cell_cluster_channel_avg():
    fovs = ['fov1', 'fov2']
    chans = ['chan1', 'chan2', 'chan3']

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: no channel average file provided
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.compute_cell_cluster_channel_avg(
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
        weighted_cell_table.to_csv(
            os.path.join(temp_dir, 'weighted_cell_channel.csv'),
            index=False
        )

        # error check: bad cell_cluster_col provided
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_cluster_channel_avg(
                fovs, chans, temp_dir, 'weighted_cell_channel.csv',
                'cell_consensus', cell_cluster_col='bad_cluster_col'
            )

        # create a dummy cell consensus data file
        # the actual column prefix won't matter for this test
        consensus_data = pd.DataFrame(
            np.random.randint(0, 100, (10, 3)),
            columns=['pixel_meta_cluster_rename_%s' % str(i) for i in np.arange(3)]
        )

        # assign dummy cell cluster labels
        consensus_data['cell_som_cluster'] = np.repeat(np.arange(5), 2)

        # assign dummy consensus cluster labels
        consensus_data['cell_meta_cluster'] = np.repeat(np.arange(2), 5)

        # assign the same FOV and segmentation_label data to consensus_data
        consensus_data[['fov', 'segmentation_label']] = weighted_cell_table[
            ['fov', 'segmentation_label']
        ].copy()

        # write consensus data
        consensus_path = os.path.join(temp_dir, 'cell_mat_consensus.feather')
        feather.write_dataframe(consensus_data, consensus_path)

        # test averages for cell SOM clusters
        cell_channel_avg = cell_cluster_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.csv',
            'cell_mat_consensus.feather', cell_cluster_col='cell_som_cluster'
        )

        # assert the same SOM clusters were assigned
        assert np.all(cell_channel_avg['cell_som_cluster'].values == np.arange(5))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_som_cluster').shape == (5, 3)

        # test averages for cell meta clusters
        cell_channel_avg = cell_cluster_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.csv',
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

        # define a pixel data directory
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_path')
        os.mkdir(pixel_data_path)

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

            fov_table['pixel_meta_cluster_rename'] = np.repeat(np.arange(2), 25)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_data_path,
                                                            fov + '.feather'))

        # iterate over both cluster col vals
        for cluster_col in ['pixel_som_cluster', 'pixel_meta_cluster_rename']:
            # count number of clusters for each cell
            cell_counts, _ = cell_cluster_utils.create_c2pc_data(
                fovs, pixel_data_path, cell_table_path, pixel_cluster_col=cluster_col
            )

            # define a sample cluster_avgs table
            num_repeats = 3 if cluster_col == 'pixel_som_cluster' else 2
            cluster_avg = pd.DataFrame(
                np.repeat([[0.1, 0.2, 0.4]], num_repeats, axis=0),
                columns=chans
            )
            cluster_labels = np.arange(num_repeats)
            cluster_avg[cluster_col] = cluster_labels

            # error check: invalid fovs provided
            with pytest.raises(ValueError):
                cell_cluster_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, fovs=['fov2', 'fov3']
                )

            # error check: invalid pixel_cluster_col provided
            with pytest.raises(ValueError):
                cell_cluster_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, pixel_cluster_col='bad_cluster_col'
                )

            # test for all and some fovs
            for fov_list in [None, fovs[:1]]:
                # test with som cluster counts and all fovs
                channel_avg = cell_cluster_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, fovs=fov_list, pixel_cluster_col=cluster_col
                )

                # subset over just the marker values
                channel_avg_markers = channel_avg[chans].values

                # define the actual values, num rows will be different depending on fov_list
                if fov_list is None:
                    num_repeats = 10
                else:
                    num_repeats = 5

                actual_markers = np.tile(
                    np.array([0.2, 0.4, 0.8]), num_repeats
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
            cell_cluster_utils.create_c2pc_data(
                fovs, 'consensus', 'cell_table', pixel_cluster_col='bad_col'
            )

        # write cell table
        cell_table_path = os.path.join(temp_dir, 'cell_table_size_normalized.csv')
        cell_table.to_csv(cell_table_path, index=False)

        # define a pixel data directory
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_path')
        os.mkdir(pixel_data_path)

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

            fov_table['pixel_meta_cluster_rename'] = np.repeat(np.arange(2), 25)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_data_path,
                                                            fov + '.feather'))

        # error check: not all required columns provided in cell table
        with pytest.raises(ValueError):
            bad_cell_table = cell_table.copy()
            bad_cell_table = bad_cell_table.rename({'cell_size': 'bad_col'}, axis=1)
            bad_cell_table_path = os.path.join(temp_dir, 'bad_cell_table.csv')
            bad_cell_table.to_csv(bad_cell_table_path, index=False)

            cluster_counts, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
                fovs, pixel_data_path, bad_cell_table_path,
                pixel_cluster_col='pixel_som_cluster'
            )

        # test counts on the pixel cluster column
        cluster_counts, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, pixel_cluster_col='pixel_som_cluster'
        )

        # assert we actually created the cluster_cols
        som_cluster_cols = ['pixel_som_cluster_' + str(cluster_num)
                            for cluster_num in np.arange(3)]
        misc_utils.verify_in_list(
            cluster_id_cols=som_cluster_cols,
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
            np.equal(np.array(correct_val), cluster_counts[som_cluster_cols].values)
        )
        assert np.all(
            np.equal(np.array(correct_val) / 5, cluster_counts_norm[som_cluster_cols].values)
        )

        # test counts on the consensus cluster column
        cluster_counts, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path,
            pixel_cluster_col='pixel_meta_cluster_rename'
        )

        # assert we actually created the pixel_meta_cluster_rename_ cols
        meta_cluster_cols = ['pixel_meta_cluster_rename_' + str(cluster_num)
                             for cluster_num in np.arange(2)]
        misc_utils.verify_in_list(
            hCluster_id_cols=meta_cluster_cols,
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
            np.equal(np.array(correct_val), cluster_counts[meta_cluster_cols].values)
        )
        assert np.all(
            np.equal(np.array(correct_val) / 5, cluster_counts_norm[meta_cluster_cols].values)
        )


def test_train_cell_som(mocker):
    # basic error check: bad path to cell table path
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.train_cell_som(
                fovs=['fov0'], channels=['chan0'], base_dir=temp_dir,
                pixel_data_dir='data_dir', cell_table_path='bad_cell_table.csv'
            )

    # basic error check: bad path to pixel data dir
    with tempfile.TemporaryDirectory() as temp_dir:
        blank_cell_table = pd.DataFrame()
        blank_cell_table.to_csv(
            os.path.join(temp_dir, 'sample_cell_table.csv'),
            index=False
        )

        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.train_cell_som(
                fovs=['fov0'], channels=['chan0'], base_dir=temp_dir,
                pixel_data_dir='data_dir',
                cell_table_path=os.path.join(temp_dir, 'sample_cell_table.csv')
            )

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

        # define a pixel data directory with SOM and meta cluster labels
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_dir')
        os.mkdir(pixel_data_path)

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

            fov_table['pixel_meta_cluster_rename'] = np.repeat(np.arange(2), 500)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_data_path,
                                                            fov + '.feather'))

        # bad cluster_col provided
        with pytest.raises(ValueError):
            cell_cluster_utils.train_cell_som(
                fovs, chan_list, temp_dir, 'pixel_data_dir', cell_table_path,
                pixel_cluster_col='bad_cluster'
            )

        # TEST 1: computing weights using pixel clusters
        # compute cluster counts
        _, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_som_cluster'
        )

        # write cluster count
        cluster_counts_norm_path = os.path.join(temp_dir, 'cluster_counts_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_norm_path)

        # add mocked function to "train_cell_som"
        mocker.patch(
            'ark.phenotyping.cell_cluster_utils.train_cell_som',
            mocked_train_cell_som
        )

        # "train" the cell SOM using mocked function
        cell_cluster_utils.train_cell_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir,
            pixel_data_dir='pixel_data_dir',
            cell_table_path=cell_table_path,
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
        _, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_meta_cluster_rename'
        )

        # write cluster count
        cluster_counts_norm_path = os.path.join(temp_dir, 'cluster_counts_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_norm_path)

        # add mocked function to "train" cell SOM
        mocker.patch(
            'ark.phenotyping.cell_cluster_utils.train_cell_som',
            mocked_train_cell_som
        )

        # "train" the cell SOM using mocked function
        cell_cluster_utils.train_cell_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir,
            pixel_data_dir='pixel_data_dir',
            cell_table_path=cell_table_path,
            pixel_cluster_col='pixel_meta_cluster_rename'
        )

        # assert cell weights has been created
        assert os.path.exists(os.path.join(temp_dir, 'cell_weights.feather'))

        # read in the cell weights
        cell_weights = feather.read_dataframe(os.path.join(temp_dir, 'cell_weights.feather'))

        # assert we created the columns needed
        misc_utils.verify_same_elements(
            cluster_col_labels=['pixel_meta_cluster_rename_' + str(i) for i in range(2)],
            cluster_weights_names=cell_weights.columns.values
        )

        # assert the shape
        assert cell_weights.shape == (100, 2)


def test_cluster_cells(mocker):
    # basic error check: path to cell counts norm does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.cluster_cells(
                base_dir=temp_dir, cluster_counts_norm_name='bad_path'
            )

    # basic error check: path to cell weights does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy cluster_counts_norm_name file
        cluster_counts_norm = pd.DataFrame()
        cluster_counts_norm.to_csv(
            os.path.join(temp_dir, 'cluster_counts_norm.feather'),
            index=False
        )

        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.cluster_cells(base_dir=temp_dir,
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
            cell_cluster_utils.cluster_cells(base_dir=temp_dir)

        # generate a random weights matrix
        weights = pd.DataFrame(np.random.rand(100, 3), columns=cluster_cols)

        # write weights
        cell_weights_path = os.path.join(temp_dir, 'cell_weights.feather')
        feather.write_dataframe(weights, cell_weights_path)

        # bad cluster_col provided
        with pytest.raises(ValueError):
            cell_cluster_utils.cluster_cells(
                base_dir=temp_dir,
                pixel_cluster_col_prefix='bad_cluster'
            )

        # add mocked function to "cluster" cells
        mocker.patch(
            'ark.phenotyping.cell_cluster_utils.cluster_cells',
            mocked_cluster_cells
        )

        # "cluster" the cells
        cell_cluster_utils.cluster_cells(base_dir=temp_dir)

        # assert the clustered feather file has been created
        assert os.path.exists(os.path.join(temp_dir, 'cell_mat_clustered.feather'))

        # assert we didn't assign any cluster 100 or above
        cell_clustered_data = feather.read_dataframe(
            os.path.join(temp_dir, 'cell_mat_clustered.feather')
        )

        cluster_ids = cell_clustered_data['cell_som_cluster']
        assert np.all(cluster_ids < 100)


def test_cell_consensus_cluster(mocker):
    # basic error check: path to cell data does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir,
                cell_data_name='bad_path', pixel_cluster_col='blah'
            )

    # basic error check: cell cluster avg table not found
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_data = pd.DataFrame()
            feather.write_dataframe(
                cell_cluster_data, os.path.join(temp_dir, 'cell_mat.feather')
            )

            cell_cluster_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col='blah'
            )

    # basic error check: weighted channel avg table not found
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            cell_cluster_data = pd.DataFrame()
            cell_cluster_avg_data = pd.DataFrame()
            feather.write_dataframe(
                cell_cluster_data, os.path.join(temp_dir, 'cell_mat.feather')
            )
            cell_cluster_avg_data.to_csv(
                os.path.join(temp_dir, 'cell_som_cluster_avgs.csv'),
                index=False
            )

            cell_cluster_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col='blah'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # test for both pixel SOM and meta cluster column names
        for cluster_prefix in ['pixel_som_cluster', 'pixel_meta_cluster_rename']:
            # create a dummy cluster_data file
            cluster_data = pd.DataFrame(
                np.random.randint(0, 100, (100, 3)),
                columns=['%s_%d' % (cluster_prefix, i) for i in np.arange(3)]
            )

            # assign dummy cell cluster labels
            cluster_data['cell_som_cluster'] = np.repeat(np.arange(10), 10)

            # write clustered data
            clustered_path = os.path.join(temp_dir, 'cell_mat.feather')
            feather.write_dataframe(cluster_data, clustered_path)

            # compute average counts of each pixel SOM/meta cluster across all cell SOM clusters
            cluster_avg = cell_cluster_utils.compute_cell_cluster_count_avg(
                clustered_path, pixel_cluster_col_prefix=cluster_prefix,
                cell_cluster_col='cell_som_cluster'
            )

            # write cluster average
            cluster_avg_path = os.path.join(temp_dir, 'cell_som_cluster_avgs.csv')
            cluster_avg.to_csv(cluster_avg_path, index=False)

            # create a dummy weighted channel average table
            weighted_cell_table = pd.DataFrame()

            # write dummy weighted channel average table
            weighted_cell_path = os.path.join(temp_dir, 'weighted_cell_table.csv')
            weighted_cell_table.to_csv(weighted_cell_path, index=False)

            # add mocked function to "consensus cluster" cell average data
            mocker.patch(
                'ark.phenotyping.cell_cluster_utils.cell_consensus_cluster',
                mocked_cell_consensus_cluster
            )

            # "consensus cluster" the cells
            cell_cluster_utils.cell_consensus_cluster(
                fovs=[], channels=[], base_dir=temp_dir, pixel_cluster_col=cluster_prefix
            )

            cell_consensus_data = feather.read_dataframe(
                os.path.join(temp_dir, 'cell_mat.feather')
            )

            # assert the cell_som_cluster labels are intact
            assert np.all(
                cluster_data['cell_som_cluster'].values ==
                cell_consensus_data['cell_som_cluster'].values
            )

            # assert we idn't assign any cluster 2 or above
            cluster_ids = cell_consensus_data['cell_meta_cluster']
            assert np.all(cluster_ids < 2)


def test_apply_cell_meta_cluster_remapping():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad path to pixel consensus dir
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'bad_consensus_name', 'remapped_name.csv',
                'pixel_som_cluster', 'som_count_avgs.csv', 'meta_count_avgs.csv',
                'weighted_cell_table.csv', 'som_chan_avgs.csv', 'meta_chan_avgs.csv'
            )

        # make a dummy consensus path
        cell_cluster_data = pd.DataFrame()
        feather.write_dataframe(
            cell_cluster_data, os.path.join(temp_dir, 'cell_mat_clustered.feather')
        )

        # basic error check: bad path to remapped name
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather', 'remapped_name.csv',
                'pixel_som_cluster', 'som_count_avgs.csv', 'meta_count_avgs.csv',
                'weighted_cell_table.csv', 'som_chan_avgs.csv', 'meta_chan_avgs.csv'
            )

        # make a dummy remapping
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_cell_remapping.csv'), index=False
        )

        # basic error check: bad path to cell SOM cluster pixel counts
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'pixel_som_cluster', 'som_count_avgs.csv',
                'meta_count_avgs.csv', 'weighted_cell_table.csv', 'som_chan_avgs.csv',
                'meta_chan_avgs.csv'
            )

        # make a dummy cell SOM cluster pixel counts
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_som_count_avgs.csv'), index=False
        )

        # basic error check: bad path to cell meta cluster pixel counts
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'pixel_som_cluster', 'sample_som_count_avgs.csv',
                'meta_count_avgs.csv', 'weighted_cell_table.csv', 'som_chan_avgs.csv',
                'meta_chan_avgs.csv'
            )

        # make a dummy cell meta cluster pixel counts
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_meta_count_avgs.csv'), index=False
        )

        # basic error check: bad path to weighted cell table
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'pixel_som_cluster', 'sample_som_count_avgs.csv',
                'sample_meta_count_avgs.csv', 'weighted_cell_table.csv', 'som_chan_avgs.csv',
                'meta_chan_avgs.csv'
            )

        # make a dummy weighted cell table
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_weighted_cell_table.csv'), index=False
        )

        # basic error check: bad path to cell SOM weighted channel avgs
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'pixel_som_cluster', 'sample_som_count_avgs.csv',
                'sample_meta_count_avgs.csv', 'sample_weighted_cell_table.csv',
                'som_chan_avgs.csv', 'meta_chan_avgs.csv'
            )

        # make a dummy weighted chan avg per cell SOM table
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_som_chan_avgs.csv'), index=False
        )

        # basic error check: bad path to cell meta weighted channel avgs
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'pixel_som_cluster', 'sample_som_count_avgs.csv',
                'sample_meta_count_avgs.csv', 'sample_weighted_cell_table.csv',
                'sample_som_chan_avgs.csv', 'meta_chan_avgs.csv'
            )

        # make a dummy weighted chan avg per cell meta table
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_meta_chan_avgs.csv'), index=False
        )

        # basic error check: bad pixel cluster col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cell_mat_clustered.feather',
                'sample_cell_remapping.csv', 'bad_pixel_cluster', 'sample_som_count_avgs.csv',
                'sample_meta_count_avgs.csv', 'sample_weighted_cell_table.csv',
                'sample_som_chan_avgs.csv', 'sample_meta_chan_avgs.csv'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # define the pixel cluster cols
        pixel_cluster_cols = ['%s_%s' % ('pixel_meta_cluster_rename', str(i))
                              for i in np.arange(3)]

        # create a dummy cluster_data file
        # for remapping, pixel prefix (pixel_som_cluster or pixel_meta_cluster_rename) irrelevant
        cluster_data = pd.DataFrame(
            np.repeat([[1, 2, 3]], repeats=1000, axis=0),
            columns=pixel_cluster_cols
        )

        # assign dummy SOM cluster labels
        cluster_data['cell_som_cluster'] = np.repeat(np.arange(100), 10)

        # assign dummy meta cluster labels
        cluster_data['cell_meta_cluster'] = np.repeat(np.arange(10), 100)

        # assign dummy fovs
        cluster_data.loc[0:499, 'fov'] = 'fov1'
        cluster_data.loc[500:999, 'fov'] = 'fov2'

        # assign dummy segmentation labels, 50 cells for each
        cluster_data.loc[0:499, 'segmentation_label'] = np.arange(500)
        cluster_data.loc[500:999, 'segmentation_label'] = np.arange(500)

        # write clustered data
        clustered_path = os.path.join(temp_dir, 'cell_mat_consensus.feather')
        feather.write_dataframe(cluster_data, clustered_path)

        # create an example cell SOM pixel counts table
        som_pixel_counts = pd.DataFrame(
            np.repeat([[1, 2, 3]], repeats=100, axis=0),
            columns=pixel_cluster_cols
        )
        som_pixel_counts['cell_som_cluster'] = np.arange(100)
        som_pixel_counts['cell_meta_cluster'] = np.repeat(np.arange(10), 10)

        som_pixel_counts.to_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_count_avgs.csv'), index=False
        )

        # since the equivalent pixel counts table for meta clusters will be overwritten
        # just make it a blank slate
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_count_avgs.csv'), index=False
        )

        # create an example weighted cell table
        chans = ['chan0', 'chan1', 'chan2']
        weighted_cell_table = pd.DataFrame(
            np.repeat([[0.1, 0.2, 0.3]], repeats=1000, axis=0),
            columns=chans
        )

        # assign dummy fovs
        weighted_cell_table.loc[0:499, 'fov'] = 'fov1'
        weighted_cell_table.loc[500:999, 'fov'] = 'fov2'

        # assign dummy segmentation labels, 50 cells for each
        weighted_cell_table.loc[0:499, 'segmentation_label'] = np.arange(500)
        weighted_cell_table.loc[500:999, 'segmentation_label'] = np.arange(500)

        # assign dummy cell sizes, these won't really matter for this test
        weighted_cell_table['cell_size'] = 5

        # save weighted cell table
        weighted_cell_table_path = os.path.join(temp_dir, 'weighted_cell_table.csv')
        weighted_cell_table.to_csv(weighted_cell_table_path, index=False)

        # create an example cell SOM weighted channel average table
        som_weighted_chan_avg = pd.DataFrame(
            np.repeat([[0.1, 0.2, 0.3]], repeats=100, axis=0),
            columns=pixel_cluster_cols
        )
        som_weighted_chan_avg['cell_som_cluster'] = np.arange(100)
        som_weighted_chan_avg['cell_meta_cluster'] = np.repeat(np.arange(10), 10)

        som_weighted_chan_avg.to_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_chan_avgs.csv'), index=False
        )

        # since the equivalent average weighted channel table for meta clusters will be overwritten
        # just make it a blank slate
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_chan_avgs.csv'), index=False
        )

        # define a dummy remap scheme and save
        # NOTE: cell mappings don't have the same issue of having more SOM clusters defined
        # than there are in the cell table there is only one cell table (as opposed to
        # multiple pixel tabels per FOV)
        sample_cell_remapping = {
            'cluster': [i for i in np.arange(100)],
            'metacluster': [int(i / 5) for i in np.arange(100)],
            'mc_name': ['meta' + str(int(i / 5)) for i in np.arange(100)]
        }
        sample_cell_remapping = pd.DataFrame.from_dict(sample_cell_remapping)
        sample_cell_remapping.to_csv(
            os.path.join(temp_dir, 'sample_cell_remapping.csv'),
            index=False
        )

        # error check: bad columns provided in the SOM to meta cluster map csv input
        with pytest.raises(ValueError):
            bad_sample_cell_remapping = sample_cell_remapping.copy()
            bad_sample_cell_remapping = bad_sample_cell_remapping.rename(
                {'mc_name': 'bad_col'},
                axis=1
            )
            bad_sample_cell_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_cell_remapping.csv'),
                index=False
            )

            # run the remapping process
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov1', 'fov2'],
                chans,
                temp_dir,
                'cell_mat_consensus.feather',
                'bad_sample_cell_remapping.csv',
                'pixel_meta_cluster_rename',
                'sample_cell_som_cluster_count_avgs.csv',
                'sample_cell_meta_cluster_count_avgs.csv',
                'weighted_cell_table.csv',
                'sample_cell_som_cluster_chan_avgs.csv',
                'sample_cell_meta_cluster_chan_avgs.csv'
            )

        # error check: mapping does not contain every SOM label
        with pytest.raises(ValueError):
            bad_sample_cell_remapping = {
                'cluster': [1, 2],
                'metacluster': [1, 2],
                'mc_name': ['m1', 'm2']
            }
            bad_sample_cell_remapping = pd.DataFrame.from_dict(bad_sample_cell_remapping)
            bad_sample_cell_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_cell_remapping.csv'),
                index=False
            )

            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov1', 'fov2'],
                chans,
                temp_dir,
                'cell_mat_consensus.feather',
                'bad_sample_cell_remapping.csv',
                'pixel_meta_cluster_rename',
                'sample_cell_som_cluster_count_avgs.csv',
                'sample_cell_meta_cluster_count_avgs.csv',
                'weighted_cell_table.csv',
                'sample_cell_som_cluster_chan_avgs.csv',
                'sample_cell_meta_cluster_chan_avgs.csv'
            )

        # run the remapping process
        cell_cluster_utils.apply_cell_meta_cluster_remapping(
            ['fov1', 'fov2'],
            chans,
            temp_dir,
            'cell_mat_consensus.feather',
            'sample_cell_remapping.csv',
            'pixel_meta_cluster_rename',
            'sample_cell_som_cluster_count_avgs.csv',
            'sample_cell_meta_cluster_count_avgs.csv',
            'weighted_cell_table.csv',
            'sample_cell_som_cluster_chan_avgs.csv',
            'sample_cell_meta_cluster_chan_avgs.csv'
        )

        # read remapped cell data in
        remapped_cell_data = feather.read_dataframe(clustered_path)

        # assert the counts of each cell cluster is 50
        assert np.all(remapped_cell_data['cell_meta_cluster'].value_counts().values == 50)

        # used for mapping verification
        actual_som_to_meta = sample_cell_remapping[
            ['cluster', 'metacluster']
        ].drop_duplicates().sort_values(by='cluster')
        actual_meta_id_to_name = sample_cell_remapping[
            ['metacluster', 'mc_name']
        ].drop_duplicates().sort_values(by='metacluster')

        # assert the mapping is the same for cell SOM to meta cluster
        som_to_meta = remapped_cell_data[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        # NOTE: unlike pixel clustering, we test the mapping on the entire cell table
        # rather than a FOV-by-FOV basis, so no need to ensure that some metaclusters
        # don't exist in the cell table mapping
        assert np.all(som_to_meta.values == actual_som_to_meta.values)

        # asset the mapping is the same for cell meta cluster to renamed cell meta cluster
        meta_id_to_name = remapped_cell_data[
            ['cell_meta_cluster', 'cell_meta_cluster_rename']
        ].drop_duplicates().sort_values(by='cell_meta_cluster')

        assert np.all(meta_id_to_name.values == actual_meta_id_to_name.values)

        # load the re-computed average count table per cell meta cluster in
        sample_cell_meta_cluster_count_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_count_avgs.csv')
        )

        # assert the counts per pixel cluster are correct
        result = np.repeat([[1, 2, 3]], repeats=20, axis=0)
        assert np.all(sample_cell_meta_cluster_count_avg[pixel_cluster_cols].values == result)

        # assert the correct counts were added
        assert np.all(sample_cell_meta_cluster_count_avg['count'].values == 50)

        # assert the correct metacluster labels are contained
        sample_cell_meta_cluster_count_avg = sample_cell_meta_cluster_count_avg.sort_values(
            by='cell_meta_cluster'
        )
        assert np.all(sample_cell_meta_cluster_count_avg[
            'cell_meta_cluster'
        ].values == np.arange(20))
        assert np.all(sample_cell_meta_cluster_count_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.arange(20)])

        # load the re-computed weighted average weighted channel table per cell meta cluster in
        sample_cell_meta_cluster_channel_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_chan_avgs.csv')
        )

        # assert the markers data has been updated correctly
        result = np.repeat([[0.1, 0.2, 0.3]], repeats=20, axis=0)
        assert np.all(np.round(sample_cell_meta_cluster_channel_avg[chans].values, 1) == result)

        # assert the correct metacluster labels are contained
        sample_cell_meta_cluster_channel_avg = sample_cell_meta_cluster_channel_avg.sort_values(
            by='cell_meta_cluster'
        )
        assert np.all(sample_cell_meta_cluster_channel_avg[
            'cell_meta_cluster'
        ].values == np.arange(20))
        assert np.all(sample_cell_meta_cluster_channel_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.arange(20)])

        # load the average count table per cell SOM cluster in
        sample_cell_som_cluster_count_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_count_avgs.csv')
        )

        # assert the correct number of meta clusters are in and the correct number of each
        assert len(sample_cell_som_cluster_count_avg['cell_meta_cluster'].value_counts()) == 20
        assert np.all(
            sample_cell_som_cluster_count_avg['cell_meta_cluster'].value_counts().values == 5
        )

        # assert the correct metacluster labels are contained
        sample_cell_som_cluster_count_avg = sample_cell_som_cluster_count_avg.sort_values(
            by='cell_meta_cluster'
        )

        assert np.all(sample_cell_som_cluster_count_avg[
            'cell_meta_cluster'
        ].values == np.repeat(np.arange(20), repeats=5))
        assert np.all(sample_cell_som_cluster_count_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.repeat(np.arange(20), repeats=5)])

        # load the average weighted channel expression per cell SOM cluster in
        sample_cell_som_cluster_chan_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_chan_avgs.csv')
        )

        # assert the correct number of meta clusters are in and the correct number of each
        assert len(sample_cell_som_cluster_chan_avg['cell_meta_cluster'].value_counts()) == 20
        assert np.all(
            sample_cell_som_cluster_chan_avg['cell_meta_cluster'].value_counts().values == 5
        )

        # assert the correct metacluster labels are contained
        sample_cell_som_cluster_chan_avg = sample_cell_som_cluster_chan_avg.sort_values(
            by='cell_meta_cluster'
        )

        assert np.all(sample_cell_som_cluster_chan_avg[
            'cell_meta_cluster'
        ].values == np.repeat(np.arange(20), repeats=5))
        assert np.all(sample_cell_som_cluster_chan_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.repeat(np.arange(20), repeats=5)])


def test_generate_weighted_channel_avg_heatmap():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad cluster channel avgs path
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.generate_weighted_channel_avg_heatmap(
                os.path.join(temp_dir, 'bad_channel_avg.csv'),
                'cell_som_cluster', [], {}, {}
            )

        # basic error check: bad cell cluster col provided
        with pytest.raises(ValueError):
            dummy_chan_avg = pd.DataFrame().to_csv(
                os.path.join(temp_dir, 'sample_channel_avg.csv')
            )
            cell_cluster_utils.generate_weighted_channel_avg_heatmap(
                os.path.join(temp_dir, 'sample_channel_avg.csv'),
                'bad_cluster_col', [], {}, {}
            )

        # test 1: cell SOM cluster channel avg
        sample_channel_avg = pd.DataFrame(
            np.random.rand(10, 3),
            columns=['chan1', 'chan2', 'chan3']
        )

        sample_channel_avg['cell_som_cluster'] = np.arange(1, 11)
        sample_channel_avg['cell_meta_cluster'] = np.repeat(np.arange(1, 6), repeats=2)
        sample_channel_avg['cell_meta_cluster_rename'] = [
            'meta' % i for i in np.repeat(np.arange(1, 6), repeats=2)
        ]

        sample_channel_avg.to_csv(
            os.path.join(temp_dir, 'sample_channel_avg.csv')
        )

        # error check aside: bad channel names provided
        with pytest.raises(ValueError):
            cell_cluster_utils.generate_weighted_channel_avg_heatmap(
                os.path.join(temp_dir, 'sample_channel_avg.csv'),
                'cell_som_cluster', ['chan1', 'chan4'], {}, {}
            )

        # define a sample colormap (raw and renamed)
        raw_cmap = {
            1: 'red',
            2: 'blue',
            3: 'green',
            4: 'purple',
            5: 'orange'
        }

        renamed_cmap = {
            'meta1': 'red',
            'meta2': 'blue',
            'meta3': 'green',
            'meta4': 'purple',
            'meta5': 'orange'
        }

        # assert visualization runs
        cell_cluster_utils.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_som_cluster', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )

        # test 2: cell meta cluster channel avg
        sample_channel_avg = sample_channel_avg.drop(columns='cell_som_cluster')
        sample_channel_avg.to_csv(
            os.path.join(temp_dir, 'sample_channel_avg.csv')
        )

        # assert visualization runs
        cell_cluster_utils.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_meta_cluster_rename', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )


def test_add_consensus_labels_cell_table():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: cell table path does not exist
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.add_consensus_labels_cell_table(
                temp_dir, 'bad_cell_table_path', ''
            )

        # create a basic cell table
        # NOTE: randomize the rows a bit to fully test merge functionality
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['chan0', 'chan1', 'chan2']
        cell_table_data = {
            'cell_size': np.repeat(1, 300),
            'fov': np.repeat(['fov0', 'fov1', 'fov2'], 100),
            'chan0': np.random.rand(300),
            'chan1': np.random.rand(300),
            'chan2': np.random.rand(300),
            'label': np.tile(np.arange(1, 101), 3)
        }
        cell_table = pd.DataFrame.from_dict(cell_table_data)
        cell_table = shuffle(cell_table).reset_index(drop=True)
        cell_table.to_csv(os.path.join(temp_dir, 'cell_table.csv'), index=False)

        # basic error check: cell consensus data does not exist
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.add_consensus_labels_cell_table(
                temp_dir, os.path.join(temp_dir, 'cell_table.csv'), 'bad_cell_consensus_name'
            )

        cell_consensus_data = {
            'cell_size': np.repeat(1, 300),
            'fov': np.repeat(['fov0', 'fov1', 'fov2'], 100),
            'pixel_meta_cluster_rename_1': np.random.rand(300),
            'pixel_meta_cluster_rename_2': np.random.rand(300),
            'pixel_meta_cluster_rename_3': np.random.rand(300),
            'segmentation_label': np.tile(np.arange(1, 101), 3),
            'cell_som_cluster': np.tile(np.arange(1, 101), 3),
            'cell_meta_cluster': np.tile(np.arange(1, 21), 15),
            'cell_meta_cluster_rename': np.tile(
                ['cell_meta_%d' % i for i in np.arange(1, 21)], 15
            )
        }

        cell_consensus = pd.DataFrame.from_dict(cell_consensus_data)
        feather.write_dataframe(
            cell_consensus,
            os.path.join(temp_dir, 'cell_consensus.feather'),
            compression='uncompressed'
        )

        # generate the new cell table
        cell_cluster_utils.add_consensus_labels_cell_table(
            temp_dir, os.path.join(temp_dir, 'cell_table.csv'), 'cell_consensus.feather'
        )

        # assert cell_table.csv still exists
        assert os.path.exists(os.path.join(temp_dir, 'cell_table_cell_labels.csv'))

        # read in the new cell table
        cell_table_with_labels = pd.read_csv(os.path.join(temp_dir, 'cell_table_cell_labels.csv'))

        # assert cell_meta_cluster column added
        assert 'cell_meta_cluster' in cell_table_with_labels.columns.values

        # assert new cell table meta cluster labels same as rename column in consensus data
        # NOTE: make sure to sort cell table values since it was randomized to test merging
        assert np.all(
            cell_table_with_labels.sort_values(
                by=['fov', 'label']
            )['cell_meta_cluster'].values == cell_consensus['cell_meta_cluster_rename'].values
        )

        # now test a cell table that has more cells than usual
        cell_table_data = {
            'cell_size': np.repeat(1, 600),
            'fov': np.repeat(['fov0', 'fov1', 'fov2'], 200),
            'chan0': np.random.rand(600),
            'chan1': np.random.rand(600),
            'chan2': np.random.rand(600),
            'label': np.tile(np.arange(1, 201), 3)
        }
        cell_table = pd.DataFrame.from_dict(cell_table_data)
        cell_table = shuffle(cell_table).reset_index(drop=True)
        cell_table.to_csv(os.path.join(temp_dir, 'cell_table.csv'), index=False)

        # generate the new cell table
        cell_cluster_utils.add_consensus_labels_cell_table(
            temp_dir, os.path.join(temp_dir, 'cell_table.csv'), 'cell_consensus.feather'
        )

        # assert cell_table.csv still exists
        assert os.path.exists(os.path.join(temp_dir, 'cell_table_cell_labels.csv'))

        # read in the new cell table
        cell_table_with_labels = pd.read_csv(os.path.join(temp_dir, 'cell_table_cell_labels.csv'))

        # assert cell_meta_cluster column added
        assert 'cell_meta_cluster' in cell_table_with_labels.columns.values

        # assert that for labels 1-100 per FOV, the meta_cluster_labels are the same
        # NOTE: make sure to sort cell table values since it was randomized to test merging
        cell_table_with_labeled_cells = cell_table_with_labels[
            cell_table_with_labels['label'] <= 100
        ]
        assert np.all(
            cell_table_with_labeled_cells.sort_values(
                by=['fov', 'label']
            )['cell_meta_cluster'].values == cell_consensus['cell_meta_cluster_rename'].values
        )

        # assert that for labels 101-200 per FOV, the meta_cluster_labels are set to 'Unassigned'
        cell_table_with_unlabeled_cells = cell_table_with_labels[
            cell_table_with_labels['label'] > 100
        ]
        assert np.all(
            cell_table_with_unlabeled_cells['cell_meta_cluster'].values == 'Unassigned'
        )
