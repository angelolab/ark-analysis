import os
import tempfile

import feather
import numpy as np
import pandas as pd
import pytest
from alpineer import misc_utils

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils
import ark.phenotyping.cell_som_clustering as cell_som_clustering
import ark.phenotyping.cluster_helpers as cluster_helpers

parametrize = pytest.mark.parametrize


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

        # TEST 1: computing SOM weights using pixel clusters
        # compute cluster counts
        _, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_som_cluster'
        )

        # train the cell SOM
        cell_pysom = cell_som_clustering.train_cell_som(
            fovs=fovs,
            base_dir=temp_dir,
            cell_table_path=cell_table_path,
            cell_som_cluster_cols=['pixel_som_cluster_%d' % i for i in np.arange(15)],
            cell_som_input_data=cluster_counts_norm
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

        # remove cell weights and weighted channel average file for next test
        os.remove(cell_pysom.weights_path)

        # TEST 2: computing weights using hierarchical clusters
        _, cluster_counts_norm = cell_cluster_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_meta_cluster_rename'
        )

        # train the cell SOM
        cell_pysom = cell_som_clustering.train_cell_som(
            fovs=fovs,
            base_dir=temp_dir,
            cell_table_path=cell_table_path,
            cell_som_cluster_cols=['pixel_meta_cluster_rename_%d' % i for i in np.arange(2)],
            cell_som_input_data=cluster_counts_norm
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


# NOTE: overwrite functionality tested in cluster_helpers_test.py
@parametrize('pixel_cluster_prefix', ['pixel_som_cluster', 'pixel_meta_cluster_rename'])
@parametrize('existing_som_col', [False, True])
def test_cluster_cells(pixel_cluster_prefix, existing_som_col):
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

        if existing_som_col:
            cluster_counts['cell_som_cluster'] = -1

        # write cluster counts
        cluster_counts_path = os.path.join(temp_dir, 'cluster_counts.feather')
        feather.write_dataframe(cluster_counts, cluster_counts_path)

        # create size normalized counts
        cluster_counts_size_norm = cluster_counts.copy()
        cluster_counts_size_norm[cluster_cols] = cluster_counts_size_norm[cluster_cols] / 5

        # error test: no weights assigned to cell pysom object
        with pytest.raises(ValueError):
            cell_pysom_bad = cluster_helpers.CellSOMCluster(
                cluster_counts_size_norm, 'bad_path.feather', [-1], cluster_cols
            )

            cell_som_clustering.cluster_cells(
                base_dir=temp_dir,
                cell_pysom=cell_pysom_bad,
                cell_som_cluster_cols=cluster_cols
            )

        # generate a random SOM weights matrix
        som_weights = pd.DataFrame(np.random.rand(100, 3), columns=cluster_cols)

        # write SOM weights
        cell_som_weights_path = os.path.join(temp_dir, 'cell_som_weights.feather')
        feather.write_dataframe(som_weights, cell_som_weights_path)

        # define a CellSOMCluster object
        cell_pysom = cluster_helpers.CellSOMCluster(
            cluster_counts_size_norm, cell_som_weights_path, [-1], cluster_cols
        )

        # assign SOM clusters to the cells
        cell_data_som_labels = cell_som_clustering.cluster_cells(
            base_dir=temp_dir,
            cell_pysom=cell_pysom,
            cell_som_cluster_cols=cluster_cols
        )

        # assert we didn't assign any cluster 100 or above
        cluster_ids = cell_data_som_labels['cell_som_cluster']
        assert np.all(cluster_ids < 100)


def test_generate_som_avg_files(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the cluster column names
        cluster_cols = [f'pixel_meta_cluster_' + str(i) for i in range(3)]

        # create a sample cluster counts file
        cluster_counts = pd.DataFrame(np.random.randint(0, 100, (100, 3)),
                                      columns=cluster_cols)

        # add metadata
        cluster_counts['fov'] = -1
        cluster_counts['cell_size'] = -1
        cluster_counts['segmentation_label'] = -1

        # add dummy SOM cluster assignments
        cluster_counts['cell_som_cluster'] = np.repeat(np.arange(1, 5), repeats=25)

        # write cluster counts
        cluster_counts_path = os.path.join(temp_dir, 'cluster_counts.feather')
        feather.write_dataframe(cluster_counts, cluster_counts_path)

        # create size normalized counts
        cluster_counts_size_norm = cluster_counts.copy()
        cluster_counts_size_norm[cluster_cols] = cluster_counts_size_norm[cluster_cols] / 5

        # define a sample weights file
        weights_path = os.path.join(temp_dir, 'cell_weights.feather')
        weights = pd.DataFrame(np.random.rand(3, 3), columns=cluster_cols)
        feather.write_dataframe(weights, weights_path)

        # error test: no SOM labels assigned to cell_som_input_data
        with pytest.raises(ValueError):
            bad_cluster_counts = cluster_counts_size_norm.copy()
            bad_cluster_counts = bad_cluster_counts.drop(columns='cell_som_cluster')
            cell_som_clustering.generate_som_avg_files(
                temp_dir, bad_cluster_counts, cluster_cols, 'cell_som_cluster_count_avgs.csv'
            )

        # generate the average SOM file
        cell_som_clustering.generate_som_avg_files(
            temp_dir, cluster_counts_size_norm, cluster_cols, 'cell_som_cluster_count_avgs.csv'
        )

        # assert we created SOM avg file
        cell_som_avg_file = os.path.join(temp_dir, 'cell_som_cluster_count_avgs.csv')
        assert os.path.exists(cell_som_avg_file)

        # load in the SOM avg file, assert all clusters and counts are correct
        cell_som_avg_data = pd.read_csv(cell_som_avg_file)
        assert list(cell_som_avg_data['cell_som_cluster']) == [1, 2, 3, 4]
        assert np.all(cell_som_avg_data['count'] == 25)

        # test that process doesn't run if SOM cluster file already generated
        capsys.readouterr()

        cell_som_clustering.generate_som_avg_files(
            temp_dir, cluster_counts_size_norm, cluster_cols, 'cell_som_cluster_count_avgs.csv'
        )

        output = capsys.readouterr().out
        assert output == \
            "Already generated average expression file for each cell SOM column, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run SOM averaging with overwrite flag
        cell_som_clustering.generate_som_avg_files(
            temp_dir, cluster_counts_size_norm, cluster_cols, 'cell_som_cluster_count_avgs.csv',
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating average expression file for cell SOM clusters\n"
        assert desired_status_updates in output
