import os
import tempfile
from copy import deepcopy

import feather
import numpy as np
import pandas as pd
import pytest
from sklearn.utils import shuffle
from alpineer import misc_utils

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils
import ark.phenotyping.cluster_helpers as cluster_helpers

parametrize = pytest.mark.parametrize


def test_compute_cell_cluster_count_avg():
    # define the cluster columns
    pixel_som_clusters = ['pixel_som_cluster_%d' % i for i in np.arange(3)]
    pixel_meta_clusters = ['pixel_meta_cluster_rename_%s' % str(i) for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel_cluster_col_prefix specified
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.compute_cell_cluster_count_avg(
                'clustered_path', 'bad_cluster_col_prefix', 'cell_cluster_col', False
            )

        # error check: bad cell_cluster_col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_cluster_count_avg(
                temp_dir, 'pixel_meta_cluster', 'bad_cluster_col', False
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

        # write the data to feather
        feather.write_dataframe(
            weighted_cell_table, os.path.join(temp_dir, 'weighted_cell_channel.feather')
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
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(consensus_data, cluster_counts_size_norm_path)

        # error check: bad cell_cluster_col provided
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_cluster_channel_avg(
                fovs, chans, temp_dir, 'weighted_cell_channel.feather',
                'cluster_counts_size_norm.feather', cell_cluster_col='bad_cluster_col'
            )

        # test averages for cell SOM clusters
        cell_channel_avg = cell_cluster_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.feather',
            'cluster_counts_size_norm.feather', cell_cluster_col='cell_som_cluster'
        )

        # assert the same SOM clusters were assigned
        assert np.all(cell_channel_avg['cell_som_cluster'].values == np.arange(5))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_som_cluster').shape == (5, 3)

        # test averages for cell meta clusters
        cell_channel_avg = cell_cluster_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.feather',
            'cluster_counts_size_norm.feather', cell_cluster_col='cell_meta_cluster'
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

            cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
                fovs, pixel_data_path, bad_cell_table_path,
                pixel_cluster_col='pixel_som_cluster'
            )

        # test counts on the pixel cluster column
        cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
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
            np.equal(np.array(correct_val) / 5, cluster_counts_size_norm[som_cluster_cols].values)
        )

        # test counts on the consensus cluster column
        cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
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
            np.equal(np.array(correct_val) / 5, cluster_counts_size_norm[meta_cluster_cols].values)
        )


def test_train_cell_som():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
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

        # TEST 1: computing SOM weights using pixel clusters
        # compute cluster counts
        cluster_counts, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_som_cluster'
        )

        # write cluster counts files
        cluster_counts_path = os.path.join(temp_dir, 'cluster_counts.feather')
        feather.write_dataframe(cluster_counts, cluster_counts_path)
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_size_norm_path)

        # generate a random pixel cluster channel average file
        pc_chan_avg = pd.DataFrame(np.random.rand(15, len(chan_list)), columns=chan_list)
        pc_chan_avg['pixel_som_cluster'] = np.arange(15)
        pc_chan_avg_path = os.path.join(temp_dir, 'pc_chan_avg.csv')
        pc_chan_avg.to_csv(pc_chan_avg_path, index=False)

        # train the cell SOM
        cell_pysom = cell_cluster_utils.train_cell_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir,
            pixel_data_dir='pixel_data_dir',
            cell_table_path=cell_table_path,
            pixel_cluster_col='pixel_som_cluster'
        )

        # assert cell weights has been created
        assert os.path.exists(cell_pysom.weights_path)

        # read in the cell weights
        cell_weights = feather.read_dataframe(cell_pysom.weights_path)

        # assert we created the columns needed
        misc_utils.verify_same_elements(
            cluster_col_labels=['pixel_som_cluster_' + str(i) for i in range(15)],
            cluster_som_weights_names=cell_weights.columns.values
        )

        # assert the shape
        assert cell_weights.shape == (100, 15)

        # assert weighted cell channel average file has been created
        assert os.path.exists(pc_chan_avg_path)

        # load in weighted cell channel average file and assert columns and clusters are correct
        weighted_chan_data = pd.read_csv(pc_chan_avg_path)
        assert list(weighted_chan_data.columns.values) == chan_list + ['pixel_som_cluster']
        assert np.all(weighted_chan_data['pixel_som_cluster'].values == np.arange(15))

        # remove cell weights and weighted channel average file for next test
        os.remove(cell_pysom.weights_path)
        os.remove(pc_chan_avg_path)

        # TEST 2: computing weights using hierarchical clusters
        cluster_counts, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_meta_cluster_rename'
        )

        # write cluster counts files
        cluster_counts_path = os.path.join(temp_dir, 'cluster_counts.feather')
        feather.write_dataframe(cluster_counts, cluster_counts_path)
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(cluster_counts_norm, cluster_counts_size_norm_path)

        # generate a random pixel cluster channel average file
        pc_chan_avg = pd.DataFrame(np.random.rand(2, len(chan_list)), columns=chan_list)
        pc_chan_avg['pixel_meta_cluster_rename'] = np.arange(2)
        pc_chan_avg_path = os.path.join(temp_dir, 'pc_chan_avg.csv')
        pc_chan_avg.to_csv(pc_chan_avg_path, index=False)

        # train the cell SOM
        cell_pysom = cell_cluster_utils.train_cell_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir,
            pixel_data_dir='pixel_data_dir',
            cell_table_path=cell_table_path,
            pixel_cluster_col='pixel_meta_cluster_rename'
        )

        # assert cell weights has been created
        assert os.path.exists(cell_pysom.weights_path)

        # read in the cell weights
        cell_weights = feather.read_dataframe(cell_pysom.weights_path)

        # assert we created the columns needed
        misc_utils.verify_same_elements(
            cluster_col_labels=['pixel_meta_cluster_rename_' + str(i) for i in range(2)],
            cluster_som_weights_names=cell_weights.columns.values
        )

        # assert the shape
        assert cell_weights.shape == (100, 2)

        # assert weighted cell channel average file has been created
        assert os.path.exists(pc_chan_avg_path)

        # load in weighted cell channel average file and assert columns and clusters are correct
        weighted_chan_data = pd.read_csv(pc_chan_avg_path)
        assert list(weighted_chan_data.columns.values) == chan_list + ['pixel_meta_cluster_rename']
        assert np.all(weighted_chan_data['pixel_meta_cluster_rename'].values == np.arange(2))


@parametrize('pixel_cluster_prefix', ['pixel_som_cluster', 'pixel_meta_cluster_rename'])
def test_cluster_cells(pixel_cluster_prefix):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the cluster column names
        cluster_cols = [f'{pixel_cluster_prefix}_' + str(i) for i in range(3)]

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

        # create size normalized counts
        cluster_counts_size_norm = cluster_counts.copy()
        cluster_counts_size_norm[cluster_cols] = cluster_counts_size_norm[cluster_cols] / 5

        # write size normalized counts
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(cluster_counts_size_norm, cluster_counts_size_norm_path)

        # error test: no weights assigned to cell pysom object
        with pytest.raises(ValueError):
            cell_pysom_bad = cluster_helpers.CellSOMCluster(
                cluster_counts_size_norm_path, 'bad_path.feather', cluster_cols
            )

            cell_cluster_utils.cluster_cells(base_dir=temp_dir, cell_pysom=cell_pysom_bad)

        # error test: column name mismatch for weights
        with pytest.raises(ValueError):
            bad_cluster_cols = cluster_cols[:]
            bad_cluster_cols[2], bad_cluster_cols[1] = bad_cluster_cols[1], bad_cluster_cols[2]

            weights = pd.DataFrame(np.random.rand(100, 3), columns=bad_cluster_cols)
            weights_path = os.path.join(temp_dir, 'cell_weights.feather')
            feather.write_dataframe(weights, weights_path)

            cell_pysom_bad = cluster_helpers.CellSOMCluster(
                cluster_counts_size_norm_path, weights_path, cluster_cols
            )

            cell_cluster_utils.cluster_cells(base_dir=temp_dir, cell_pysom=cell_pysom_bad)

        # generate a random SOM weights matrix
        som_weights = pd.DataFrame(np.random.rand(100, 3), columns=cluster_cols)

        # write SOM weights
        cell_som_weights_path = os.path.join(temp_dir, 'cell_som_weights.feather')
        feather.write_dataframe(som_weights, cell_som_weights_path)

        # define a CellSOMCluster object
        cell_pysom = cluster_helpers.CellSOMCluster(
            cluster_counts_size_norm_path, cell_som_weights_path, cluster_cols
        )

        # error test: bad cluster_col provided
        with pytest.raises(ValueError):
            cell_cluster_utils.cluster_cells(
                base_dir=temp_dir,
                cell_pysom=cell_pysom,
                pixel_cluster_col_prefix='bad_cluster'
            )

        # assign SOM clusters to the cells
        cell_cluster_utils.cluster_cells(
            base_dir=temp_dir,
            cell_pysom=cell_pysom,
            pixel_cluster_col_prefix=pixel_cluster_prefix
        )

        # load in the updated cluster file
        cell_data_som_labels = feather.read_dataframe(cell_pysom.cell_data_path)

        # assert we didn't assign any cluster 100 or above
        cluster_ids = cell_data_som_labels['cell_som_cluster']
        print(cluster_ids)
        assert np.all(cluster_ids < 100)

        # assert we created the cell som count avgs file
        cell_som_count_avg_path = os.path.join(temp_dir, 'cell_som_cluster_count_avgs.csv')
        assert os.path.exists(cell_som_count_avg_path)

        # load in cell som count avgs
        cell_som_count_avgs = pd.read_csv(cell_som_count_avg_path)

        # assert the columns are correct
        cell_som_avg_cols = ['cell_som_cluster'] + cluster_cols + ['count']
        assert list(cell_som_count_avgs.columns.values) == cell_som_avg_cols

        # assert the SOM clusters match
        misc_utils.verify_same_elements(
            cell_data_som_clusters=cluster_ids,
            cell_som_avg_clusters=cell_som_count_avgs['cell_som_cluster'].values
        )


@parametrize('pixel_cluster_prefix', ['pixel_som_cluster', 'pixel_meta_cluster_rename'])
def test_cell_consensus_cluster(pixel_cluster_prefix):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy cluster_data file
        cluster_data = pd.DataFrame(
            np.random.randint(0, 100, (1000, 3)),
            columns=['%s_%d' % (pixel_cluster_prefix, i) for i in np.arange(3)]
        )

        cluster_data['fov'] = np.repeat(['fov0', 'fov1'], repeats=500)
        cluster_data['segmentation_label'] = np.tile(np.arange(1, 501), reps=2)
        cluster_data['cell_som_cluster'] = np.repeat(np.arange(100), 10)

        # write clustered data
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(cluster_data, cluster_counts_size_norm_path)

        # compute average counts of each pixel SOM/meta cluster across all cell SOM clusters
        cluster_avg = cell_cluster_utils.compute_cell_cluster_count_avg(
            cluster_counts_size_norm_path, pixel_cluster_col_prefix=pixel_cluster_prefix,
            cell_cluster_col='cell_som_cluster'
        )

        # write cluster average
        cluster_avg_path = os.path.join(temp_dir, 'cell_som_cluster_avg.csv')
        cluster_avg.to_csv(cluster_avg_path, index=False)

        # create a dummy weighted channel average table
        weighted_cell_table = pd.DataFrame(
            np.random.rand(1000, 3), columns=['chan%d' % i for i in np.arange(3)]
        )
        weighted_cell_table['fov'] = np.repeat(['fov0', 'fov1'], repeats=500)
        weighted_cell_table['segmentation_label'] = np.tile(np.arange(1, 501), reps=2)

        # write dummy weighted channel average table
        weighted_cell_path = os.path.join(temp_dir, 'weighted_cell_channel.feather')
        feather.write_dataframe(weighted_cell_table, weighted_cell_path)

        # run consensus clustering
        cell_cc = cell_cluster_utils.cell_consensus_cluster(
            fovs=['fov0', 'fov1'],
            channels=['chan0', 'chan1', 'chan2'],
            base_dir=temp_dir,
            pixel_cluster_col=pixel_cluster_prefix
        )

        # assert we assigned a mapping, then sort
        assert cell_cc.mapping is not None
        sample_mapping = deepcopy(cell_cc.mapping)
        sample_mapping = sample_mapping.sort_values(by='cell_som_cluster')

        cell_consensus_data = feather.read_dataframe(
            os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        )

        # assert the cell_som_cluster labels are intact
        assert np.all(
            cluster_data['cell_som_cluster'].values ==
            cell_consensus_data['cell_som_cluster'].values
        )

        # assert the correct labels have been assigned
        cell_mapping = cell_consensus_data[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        assert np.all(sample_mapping.values == cell_mapping.values)

        # assert we generated a meta cluster average file, then load it in
        assert os.path.exists(os.path.join(temp_dir, 'cell_meta_cluster_avg.csv'))
        meta_cluster_avg = pd.read_csv(
            os.path.join(temp_dir, 'cell_meta_cluster_avg.csv')
        )

        # assert all the consensus labels have been assigned
        assert np.all(meta_cluster_avg['cell_meta_cluster'] == np.arange(1, 21))

        # load in the SOM cluster average file
        som_cluster_avg = pd.read_csv(
            os.path.join(temp_dir, 'cell_som_cluster_avg.csv')
        )

        # assert the correct labels have been assigned
        som_avg_mapping = som_cluster_avg[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        assert np.all(som_avg_mapping.values == sample_mapping.values)

        # assert we created an average weighted channel expression file for cell SOM clusters
        # then load it in
        assert os.path.exists(os.path.join(temp_dir, 'cell_som_cluster_channel_avg.csv'))
        weighted_cell_som_avgs = pd.read_csv(
            os.path.join(temp_dir, 'cell_som_cluster_channel_avg.csv')
        )

        # assert the correct labels have been assigned
        weighted_som_mapping = weighted_cell_som_avgs[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        assert np.all(sample_mapping.values == weighted_som_mapping.values)

        # assert we created an average weighted channel expression file for cell meta clusters
        # then load it in
        assert os.path.exists(os.path.join(temp_dir, 'cell_meta_cluster_channel_avg.csv'))
        weighted_cell_som_avgs = pd.read_csv(
            os.path.join(temp_dir, 'cell_meta_cluster_channel_avg.csv')
        )

        # assert all the consensus labels have been assigned
        assert np.all(weighted_cell_som_avgs['cell_meta_cluster'] == np.arange(1, 21))


def test_apply_cell_meta_cluster_remapping():
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
        cluster_counts_size_norm_path = os.path.join(temp_dir, 'cluster_counts_size_norm.feather')
        feather.write_dataframe(cluster_data, cluster_counts_size_norm_path)

        # create an example cell SOM pixel counts table
        som_pixel_counts = pd.DataFrame(
            np.repeat([[1, 2, 3]], repeats=100, axis=0),
            columns=pixel_cluster_cols
        )
        som_pixel_counts['cell_som_cluster'] = np.arange(100)
        som_pixel_counts['cell_meta_cluster'] = np.repeat(np.arange(10), 10)

        som_pixel_counts.to_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_count_avg.csv'), index=False
        )

        # since the equivalent pixel counts table for meta clusters will be overwritten
        # just make it a blank slate
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_count_avg.csv'), index=False
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
        weighted_cell_table_path = os.path.join(temp_dir, 'sample_weighted_cell_table.feather')
        feather.write_dataframe(weighted_cell_table, weighted_cell_table_path)

        # create an example cell SOM weighted channel average table
        som_weighted_chan_avg = pd.DataFrame(
            np.repeat([[0.1, 0.2, 0.3]], repeats=100, axis=0),
            columns=pixel_cluster_cols
        )
        som_weighted_chan_avg['cell_som_cluster'] = np.arange(100)
        som_weighted_chan_avg['cell_meta_cluster'] = np.repeat(np.arange(10), 10)

        som_weighted_chan_avg.to_csv(
            os.path.join(temp_dir, 'sample_cell_som_cluster_chan_avg.csv'), index=False
        )

        # since the equivalent average weighted channel table for meta clusters will be overwritten
        # just make it a blank slate
        pd.DataFrame().to_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_chan_avg.csv'), index=False
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

        # basic error check: bad pixel cluster col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'cluster_counts_size_norm.feather',
                'sample_cell_remapping.csv', 'bad_pixel_cluster',
                'sample_cell_som_cluster_count_avg.csv',
                'sample_cell_meta_cluster_count_avg.csv', 'sample_weighted_cell_table.feather',
                'sample_cell_som_cluster_chan_avg.csv', 'sample_cell_meta_cluster_chan_avg.csv'
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
                'cluster_counts_size_norm.feather',
                'bad_sample_cell_remapping.csv',
                'pixel_meta_cluster_rename',
                'sample_cell_som_cluster_count_avg.csv',
                'sample_cell_meta_cluster_count_avg.csv',
                'sample_weighted_cell_table.feather',
                'sample_cell_som_cluster_chan_avg.csv',
                'sample_cell_meta_cluster_chan_avg.csv'
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
                'cluster_counts_size_norm.feather',
                'bad_sample_cell_remapping.csv',
                'pixel_meta_cluster_rename',
                'sample_cell_som_cluster_count_avg.csv',
                'sample_cell_meta_cluster_count_avg.csv',
                'sample_weighted_cell_table.feather',
                'sample_cell_som_cluster_chan_avg.csv',
                'sample_cell_meta_cluster_chan_avg.csv'
            )

        # basic error check: bad pixel cluster col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.apply_cell_meta_cluster_remapping(
                ['fov1', 'fov2'],
                chans,
                temp_dir,
                'cluster_counts_size_norm.feather',
                'bad_sample_cell_remapping.csv',
                'bad_cluster_col',
                'sample_cell_som_cluster_count_avg.csv',
                'sample_cell_meta_cluster_count_avg.csv',
                'sample_weighted_cell_table.feather',
                'sample_cell_som_cluster_chan_avg.csv',
                'sample_cell_meta_cluster_chan_avg.csv'
            )

        # run the remapping process
        cell_cluster_utils.apply_cell_meta_cluster_remapping(
            ['fov1', 'fov2'],
            chans,
            temp_dir,
            'cluster_counts_size_norm.feather',
            'sample_cell_remapping.csv',
            'pixel_meta_cluster_rename',
            'sample_cell_som_cluster_count_avg.csv',
            'sample_cell_meta_cluster_count_avg.csv',
            'sample_weighted_cell_table.feather',
            'sample_cell_som_cluster_chan_avg.csv',
            'sample_cell_meta_cluster_chan_avg.csv'
        )

        # read remapped cell data in
        remapped_cell_data = feather.read_dataframe(cluster_counts_size_norm_path)

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
            os.path.join(temp_dir, 'sample_cell_meta_cluster_count_avg.csv')
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
            os.path.join(temp_dir, 'sample_cell_meta_cluster_chan_avg.csv')
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
            os.path.join(temp_dir, 'sample_cell_som_cluster_count_avg.csv')
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
            os.path.join(temp_dir, 'sample_cell_som_cluster_chan_avg.csv')
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
