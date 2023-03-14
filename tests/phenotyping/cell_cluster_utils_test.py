import os
import tempfile

import feather
import numpy as np
import pandas as pd
import pytest
from sklearn.utils import shuffle
from alpineer import misc_utils

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils


def test_compute_cell_som_cluster_cols_avg():
    # define the cluster columns
    pixel_som_clusters = ['pixel_som_cluster_%d' % i for i in np.arange(3)]
    pixel_meta_clusters = ['pixel_meta_cluster_rename_%s' % str(i) for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad cell_cluster_col specified
        with pytest.raises(ValueError):
            cell_cluster_utils.compute_cell_som_cluster_cols_avg(
                pd.DataFrame(), 'pixel_meta_cluster', 'bad_cluster_col', False
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

            # test for both keep_count settings
            for keep_count in [False, True]:
                # TEST 1: paveraged over cell SOM clusters
                # drop a certain set of columns when checking count avg values
                drop_cols = ['cell_som_cluster']
                if keep_count:
                    drop_cols.append('count')

                cell_cluster_avg = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
                    cluster_data, cluster_col_arr[i], 'cell_som_cluster', keep_count=keep_count
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

                cell_cluster_avg = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
                    cluster_data, cluster_col_arr[i], 'cell_meta_cluster', keep_count=keep_count
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
        correct_val_som = [[10, 0, 0],
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
            np.equal(
                np.array(correct_val_som),
                cluster_counts[som_cluster_cols].values
            )
        )
        assert np.all(
            np.equal(
                np.array(correct_val_som) / 5,
                cluster_counts_size_norm[som_cluster_cols].values
            )
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
        correct_val_meta = [[10, 0],
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
            np.equal(
                np.array(correct_val_meta),
                cluster_counts[meta_cluster_cols].values
            )
        )
        assert np.all(
            np.equal(
                np.array(correct_val_meta) / 5,
                cluster_counts_size_norm[meta_cluster_cols].values
            )
        )

        # create new FOVs that has some cluster labels that aren't in the cell table
        for fov in ['fov3', 'fov4']:
            # assume each label has 10 pixels, create dummy data for each of them
            fov_table = pd.DataFrame(np.random.rand(50, 3), columns=chans)

            # assign the fovs and labels
            fov_table['fov'] = fov
            fov_table['segmentation_label'] = np.repeat(np.arange(10), 5)

            fov_table['pixel_som_cluster'] = np.repeat(np.arange(5), 10)
            fov_table['pixel_meta_cluster_rename'] = np.repeat(np.arange(5), 10)

            # write fov data to feather
            feather.write_dataframe(fov_table, os.path.join(pixel_data_path,
                                                            fov + '.feather'))

        # append fov3 and fov4 to the cell table
        fovs += ['fov3', 'fov4']

        cell_table_34 = cell_table.copy()
        cell_table_34.loc[0:4, 'fov'] = 'fov3'
        cell_table_34.loc[5:9, 'fov'] = 'fov4'
        cell_table = pd.concat([cell_table, cell_table_34])
        cell_table.to_csv(cell_table_path, index=False)

        # test NaN counts on the SOM cluster column
        with pytest.warns(match='Pixel clusters pixel_som_cluster_3'):
            cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
                fovs, pixel_data_path, cell_table_path,
                pixel_cluster_col='pixel_som_cluster'
            )

            correct_val_som = [[10, 0, 0],
                               [10, 0, 0],
                               [5, 5, 0],
                               [0, 10, 0],
                               [0, 10, 0],
                               [0, 10, 0],
                               [0, 10, 0],
                               [0, 5, 5],
                               [0, 0, 10],
                               [0, 0, 10],
                               [5, 0, 0],
                               [5, 0, 0],
                               [0, 5, 0],
                               [0, 5, 0],
                               [0, 0, 5],
                               [5, 0, 0],
                               [5, 0, 0],
                               [0, 5, 0],
                               [0, 5, 0],
                               [0, 0, 5]]

            assert np.all(
                np.equal(
                    np.array(correct_val_som),
                    cluster_counts[som_cluster_cols].values
                )
            )
            assert np.all(
                np.equal(
                    np.array(correct_val_som) / 5,
                    cluster_counts_size_norm[som_cluster_cols].values
                )
            )

        cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path,
            pixel_cluster_col='pixel_meta_cluster_rename'
        )

        # test NaN counts on the meta cluster column
        with pytest.warns(match='Pixel clusters pixel_meta_cluster_rename_3'):
            cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
                fovs, pixel_data_path, cell_table_path,
                pixel_cluster_col='pixel_meta_cluster_rename'
            )

            correct_val_meta = [[10, 0, 0],
                                [10, 0, 0],
                                [5, 5, 0],
                                [0, 10, 0],
                                [0, 10, 0],
                                [10, 0, 0],
                                [10, 0, 0],
                                [5, 5, 0],
                                [0, 10, 0],
                                [0, 10, 0],
                                [5, 0, 0],
                                [5, 0, 0],
                                [0, 5, 0],
                                [0, 5, 0],
                                [0, 0, 5],
                                [5, 0, 0],
                                [5, 0, 0],
                                [0, 5, 0],
                                [0, 5, 0],
                                [0, 0, 5]]

            # creation of data for this step added another meta clustering column
            meta_cluster_cols += ['pixel_meta_cluster_rename_2']

            assert np.all(
                np.equal(
                    np.array(correct_val_meta),
                    cluster_counts[meta_cluster_cols].values
                )
            )
            assert np.all(
                np.equal(
                    np.array(correct_val_meta) / 5,
                    cluster_counts_size_norm[meta_cluster_cols].values
                )
            )


def test_add_consensus_labels_cell_table():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: cell table path does not exist
        with pytest.raises(FileNotFoundError):
            cell_cluster_utils.add_consensus_labels_cell_table(
                temp_dir, 'bad_cell_table_path', pd.DataFrame()
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

        # generate the new cell table
        cell_cluster_utils.add_consensus_labels_cell_table(
            temp_dir, os.path.join(temp_dir, 'cell_table.csv'), cell_consensus
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
            temp_dir, os.path.join(temp_dir, 'cell_table.csv'), cell_consensus
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
