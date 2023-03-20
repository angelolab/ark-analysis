import os
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils
import ark.phenotyping.cell_meta_clustering as cell_meta_clustering
import ark.phenotyping.cluster_helpers as cluster_helpers

parametrize = pytest.mark.parametrize


@parametrize('pixel_cluster_prefix', ['pixel_som_cluster', 'pixel_meta_cluster_rename'])
def test_cell_consensus_cluster(pixel_cluster_prefix):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the cluster column names
        cluster_cols = [f'{pixel_cluster_prefix}_' + str(i) for i in range(3)]

        # create a dummy cluster_data file
        cluster_data = pd.DataFrame(
            np.random.randint(0, 100, (1000, 3)),
            columns=cluster_cols
        )

        cluster_data['fov'] = np.repeat(['fov0', 'fov1'], repeats=500)
        cluster_data['segmentation_label'] = np.tile(np.arange(1, 501), reps=2)
        cluster_data['cell_som_cluster'] = np.repeat(np.arange(100), 10)

        # compute average values of all cluster_cols for cell SOM clusters
        cluster_avg = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
            cluster_data, cell_som_cluster_cols=cluster_cols,
            cell_cluster_col='cell_som_cluster'
        )

        # write cluster average
        cluster_avg_path = os.path.join(temp_dir, 'cell_som_cluster_avg.csv')
        cluster_avg.to_csv(cluster_avg_path, index=False)

        # run consensus clustering
        cell_cc, cell_consensus_data = cell_meta_clustering.cell_consensus_cluster(
            base_dir=temp_dir,
            cell_som_cluster_cols=cluster_cols,
            cell_som_input_data=cluster_data,
            cell_som_expr_col_avg_name='cell_som_cluster_avg.csv'
        )

        # assert we assigned a mapping, then sort
        assert cell_cc.mapping is not None
        sample_mapping = deepcopy(cell_cc.mapping)
        sample_mapping = sample_mapping.sort_values(by='cell_som_cluster')

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


def test_generate_meta_avg_files(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the cluster column names
        cluster_cols = [f'pixel_meta_cluster_' + str(i) for i in range(3)]

        # create a dummy cluster_data file
        cluster_data = pd.DataFrame(
            np.random.randint(0, 100, (1000, 3)),
            columns=cluster_cols
        )

        cluster_data['fov'] = np.repeat(['fov0', 'fov1'], repeats=500)
        cluster_data['segmentation_label'] = np.tile(np.arange(1, 501), reps=2)
        cluster_data['cell_som_cluster'] = np.repeat(np.arange(100), 10)
        cluster_data['cell_meta_cluster'] = np.repeat(np.arange(20), 50)

        # compute average values of all cluster_cols for cell SOM clusters
        cluster_avg = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
            cluster_data, cell_som_cluster_cols=cluster_cols,
            cell_cluster_col='cell_som_cluster'
        )

        # write cluster average
        cluster_avg_path = os.path.join(temp_dir, 'cell_som_cluster_avg.csv')
        cluster_avg.to_csv(cluster_avg_path, index=False)

        # define a sample SOM to meta cluster map
        som_to_meta_data = {
            'cell_som_cluster': np.arange(100),
            'cell_meta_cluster': np.repeat(np.arange(20), 5)
        }
        som_to_meta_data = pd.DataFrame.from_dict(som_to_meta_data)

        # define a sample ConsensusCluster object
        # define a dummy input file for data, we won't need it for expression average testing
        consensus_dummy_file = os.path.join(temp_dir, 'dummy_consensus_input.csv')
        pd.DataFrame().to_csv(consensus_dummy_file)
        cell_cc = cluster_helpers.PixieConsensusCluster(
            'cell', consensus_dummy_file, cluster_cols, max_k=3
        )
        cell_cc.mapping = som_to_meta_data

        # error test: no meta labels assigned to cell_som_input_data
        with pytest.raises(ValueError):
            bad_cluster_data = cluster_data.copy()
            bad_cluster_data = bad_cluster_data.drop(columns='cell_meta_cluster')
            cell_meta_clustering.generate_meta_avg_files(
                temp_dir, cell_cc, cluster_cols,
                bad_cluster_data,
                'cell_som_cluster_avg.csv',
                'cell_meta_cluster_avg.csv'
            )

        # generate the cell meta cluster file generation
        cell_meta_clustering.generate_meta_avg_files(
            temp_dir, cell_cc, cluster_cols,
            cluster_data,
            'cell_som_cluster_avg.csv',
            'cell_meta_cluster_avg.csv'
        )

        # assert we generated a meta cluster average file, then load it in
        assert os.path.exists(os.path.join(temp_dir, 'cell_meta_cluster_avg.csv'))
        meta_cluster_avg = pd.read_csv(
            os.path.join(temp_dir, 'cell_meta_cluster_avg.csv')
        )

        # assert all the consensus labels have been assigned
        assert np.all(meta_cluster_avg['cell_meta_cluster'] == np.arange(20))

        # load in the SOM cluster average file
        som_cluster_avg = pd.read_csv(
            os.path.join(temp_dir, 'cell_som_cluster_avg.csv')
        )

        # assert the correct labels have been assigned
        som_avg_mapping = som_cluster_avg[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        sample_mapping = deepcopy(cell_cc.mapping)
        sample_mapping = sample_mapping.sort_values(by='cell_som_cluster')
        assert np.all(som_avg_mapping.values == sample_mapping.values)

        # test that process doesn't run if meta cluster count avg file already generated
        capsys.readouterr()

        cell_meta_clustering.generate_meta_avg_files(
            temp_dir, cell_cc, cluster_cols,
            cluster_data,
            'cell_som_cluster_avg.csv',
            'cell_meta_cluster_avg.csv'
        )

        output = capsys.readouterr().out
        assert output == \
            "Already generated average expression file for cell meta clusters, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run meta averaging with overwrite flag
        cell_meta_clustering.generate_meta_avg_files(
            temp_dir, cell_cc, cluster_cols,
            cluster_data,
            'cell_som_cluster_avg.csv',
            'cell_meta_cluster_avg.csv',
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating average expression file for cell meta clusters\n"
        assert desired_status_updates in output


@parametrize('weighted_cell_channel_exists', [True, False])
def test_apply_cell_meta_cluster_remapping(weighted_cell_channel_exists):
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

        # define a dummy remap scheme and save
        # NOTE: cell mappings don't have the same issue of having more SOM clusters defined
        # than there are in the cell table there is only one cell table (as opposed to
        # multiple pixel tabels per FOV)
        sample_cell_remapping = {
            'cell_som_cluster': [i for i in np.arange(100)],
            'cell_meta_cluster': [int(i / 5) for i in np.arange(100)],
            'cell_meta_cluster_rename': ['meta' + str(int(i / 5)) for i in np.arange(100)]
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
                {'cell_meta_cluster_rename': 'bad_col'},
                axis=1
            )
            bad_sample_cell_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_cell_remapping.csv'),
                index=False
            )

            cell_meta_clustering.apply_cell_meta_cluster_remapping(
                temp_dir,
                cluster_data,
                'bad_sample_cell_remapping.csv'
            )

        # error check: mapping does not contain every SOM label
        with pytest.raises(ValueError):
            bad_sample_cell_remapping = {
                'cell_som_cluster': [1, 2],
                'cell_meta_cluster': [1, 2],
                'cell_meta_cluster_rename': ['m1', 'm2']
            }
            bad_sample_cell_remapping = pd.DataFrame.from_dict(bad_sample_cell_remapping)
            bad_sample_cell_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_cell_remapping.csv'),
                index=False
            )

            cell_meta_clustering.apply_cell_meta_cluster_remapping(
                temp_dir,
                cluster_data,
                'bad_sample_cell_remapping.csv'
            )

        # run the remapping process
        remapped_cell_data = cell_meta_clustering.apply_cell_meta_cluster_remapping(
            temp_dir,
            cluster_data,
            'sample_cell_remapping.csv',
        )

        # assert the counts of each cell cluster is 50
        assert np.all(remapped_cell_data['cell_meta_cluster'].value_counts().values == 50)

        # used for mapping verification
        actual_som_to_meta = sample_cell_remapping[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')
        actual_meta_id_to_name = sample_cell_remapping[
            ['cell_meta_cluster', 'cell_meta_cluster_rename']
        ].drop_duplicates().sort_values(by='cell_meta_cluster')

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


def test_generate_remap_avg_count_files():
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

        # define a dummy remap scheme and save
        # NOTE: cell mappings don't have the same issue of having more SOM clusters defined
        # than there are in the cell table there is only one cell table (as opposed to
        # multiple pixel tabels per FOV)
        sample_cell_remapping = {
            'cell_som_cluster': [i for i in np.arange(100)],
            'cell_meta_cluster': [int(i / 5) for i in np.arange(100)],
            'cell_meta_cluster_rename': ['meta' + str(int(i / 5)) for i in np.arange(100)]
        }
        sample_cell_remapping = pd.DataFrame.from_dict(sample_cell_remapping)
        sample_cell_remapping.to_csv(
            os.path.join(temp_dir, 'sample_cell_remapping.csv'),
            index=False
        )

        cell_meta_clustering.generate_remap_avg_count_files(
            temp_dir,
            cluster_data,
            'sample_cell_remapping.csv',
            pixel_cluster_cols,
            'sample_cell_som_cluster_count_avg.csv',
            'sample_cell_meta_cluster_count_avg.csv',
        )

        # load the re-computed average count table per cell meta cluster in
        sample_cell_meta_cluster_count_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_count_avg.csv')
        )

        # assert the counts per pixel cluster are correct
        result = np.repeat([[1, 2, 3]], repeats=10, axis=0)
        assert np.all(sample_cell_meta_cluster_count_avg[pixel_cluster_cols].values == result)

        # assert the correct counts were added
        assert np.all(sample_cell_meta_cluster_count_avg['count'].values == 100)

        # assert the correct metacluster labels are contained
        sample_cell_meta_cluster_count_avg = sample_cell_meta_cluster_count_avg.sort_values(
            by='cell_meta_cluster'
        )
        assert np.all(sample_cell_meta_cluster_count_avg[
            'cell_meta_cluster'
        ].values == np.arange(10))
        assert np.all(sample_cell_meta_cluster_count_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.arange(10)])

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
