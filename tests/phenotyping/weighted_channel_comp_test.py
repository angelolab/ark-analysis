import os
import tempfile
from copy import deepcopy

import feather
import numpy as np
import pandas as pd
import pytest

import ark.phenotyping.cell_cluster_utils as cell_cluster_utils
import ark.phenotyping.weighted_channel_comp as weighted_channel_comp
import ark.phenotyping.cluster_helpers as cluster_helpers


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
                weighted_channel_comp.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, fovs=['fov2', 'fov3']
                )

            # error check: invalid pixel_cluster_col provided
            with pytest.raises(ValueError):
                weighted_channel_comp.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, pixel_cluster_col='bad_cluster_col'
                )

            # test for all and some fovs
            for fov_list in [None, fovs[:1]]:
                # test with som cluster counts and all fovs
                channel_avg = weighted_channel_comp.compute_p2c_weighted_channel_avg(
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

            # test for mismatched pixel cluster columns (happens if zero-columns filtered out)
            cell_counts_trim = cell_counts.drop(columns=[f'{cluster_col}_1'])

            channel_avg = weighted_channel_comp.compute_p2c_weighted_channel_avg(
                cluster_avg, chans, cell_counts_trim, fovs=fov_list, pixel_cluster_col=cluster_col
            )

            # subset over just the marker values
            channel_avg_markers = channel_avg[chans].values

            actual_markers = np.array(
                [[0.2, 0.4, 0.8],
                 [0.2, 0.4, 0.8],
                 [0.1, 0.2, 0.4],
                 [0, 0, 0],
                 [0, 0, 0]]
            )

            # assert the values are close enough
            assert np.allclose(channel_avg_markers, actual_markers)


def test_compute_cell_cluster_weighted_channel_avg():
    fovs = ['fov1', 'fov2']
    chans = ['chan1', 'chan2', 'chan3']

    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: no channel average file provided
        with pytest.raises(FileNotFoundError):
            weighted_channel_comp.compute_cell_cluster_weighted_channel_avg(
                fovs, chans, temp_dir, 'bad_cell_table', pd.DataFrame(), 'bad_cluster_col'
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

        # error check: bad cell_cluster_col provided
        with pytest.raises(ValueError):
            weighted_channel_comp.compute_cell_cluster_weighted_channel_avg(
                fovs, chans, temp_dir, 'weighted_cell_channel.feather',
                consensus_data, cell_cluster_col='bad_cluster_col'
            )

        # test averages for cell SOM clusters
        cell_channel_avg = weighted_channel_comp.compute_cell_cluster_weighted_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.feather',
            consensus_data, cell_cluster_col='cell_som_cluster'
        )

        # assert the same SOM clusters were assigned
        assert np.all(cell_channel_avg['cell_som_cluster'].values == np.arange(5))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_som_cluster').shape == (5, 3)

        # test averages for cell meta clusters
        cell_channel_avg = weighted_channel_comp.compute_cell_cluster_weighted_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.feather',
            consensus_data, cell_cluster_col='cell_meta_cluster'
        )

        # assert the same meta clusters were assigned
        assert np.all(cell_channel_avg['cell_meta_cluster'].values == np.arange(2))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_meta_cluster').shape == (2, 3)


def test_generate_wc_avg_files(capsys):
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

        # create a dummy weighted channel average table
        weighted_cell_table = pd.DataFrame(
            np.random.rand(1000, 3), columns=['chan%d' % i for i in np.arange(3)]
        )
        weighted_cell_table['fov'] = np.repeat(['fov0', 'fov1'], repeats=500)
        weighted_cell_table['segmentation_label'] = np.tile(np.arange(1, 501), reps=2)

        # write dummy weighted channel average table
        weighted_cell_path = os.path.join(temp_dir, 'weighted_cell_channel.feather')
        feather.write_dataframe(weighted_cell_table, weighted_cell_path)

        # define a sample SOM to meta cluster map
        som_to_meta_data = {
            'cell_som_cluster': np.arange(100),
            'cell_meta_cluster': np.repeat(np.arange(20), 5)
        }
        som_to_meta_data = pd.DataFrame.from_dict(som_to_meta_data)

        # define a sample ConsensusCluster object
        # define a dummy input file for data, we won't need it for weighted channel average testing
        consensus_dummy_file = os.path.join(temp_dir, 'dummy_consensus_input.csv')
        pd.DataFrame().to_csv(consensus_dummy_file)
        cell_cc = cluster_helpers.PixieConsensusCluster(
            'cell', consensus_dummy_file, cluster_cols, max_k=3
        )
        cell_cc.mapping = som_to_meta_data

        # generate the weighted channel average file generation
        weighted_channel_comp.generate_wc_avg_files(
            fovs=['fov0', 'fov1'],
            channels=['chan0', 'chan1', 'chan2'],
            base_dir=temp_dir,
            cell_cc=cell_cc,
            cell_som_input_data=cluster_data
        )

        assert os.path.exists(os.path.join(temp_dir, 'cell_som_cluster_channel_avg.csv'))
        weighted_cell_som_avgs = pd.read_csv(
            os.path.join(temp_dir, 'cell_som_cluster_channel_avg.csv')
        )

        # assert the correct labels have been assigned
        weighted_som_mapping = weighted_cell_som_avgs[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].drop_duplicates().sort_values(by='cell_som_cluster')

        sample_mapping = deepcopy(cell_cc.mapping)
        sample_mapping = sample_mapping.sort_values(by='cell_som_cluster')
        assert np.all(sample_mapping.values == weighted_som_mapping.values)

        # assert we created an average weighted channel expression file for cell meta clusters
        # then load it in
        assert os.path.exists(os.path.join(temp_dir, 'cell_meta_cluster_channel_avg.csv'))
        weighted_cell_som_avgs = pd.read_csv(
            os.path.join(temp_dir, 'cell_meta_cluster_channel_avg.csv')
        )

        # assert all the consensus labels have been assigned
        assert np.all(weighted_cell_som_avgs['cell_meta_cluster'] == np.arange(20))

        # test that process doesn't run if weighted channel avg files already generated
        capsys.readouterr()

        weighted_channel_comp.generate_wc_avg_files(
            fovs=['fov0', 'fov1'],
            channels=['chan0', 'chan1', 'chan2'],
            base_dir=temp_dir,
            cell_cc=cell_cc,
            cell_som_input_data=cluster_data
        )

        output = capsys.readouterr().out
        assert output == \
            "Already generated average weighted channel expression files, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run weighted channel averaging with overwrite flag
        weighted_channel_comp.generate_wc_avg_files(
            fovs=['fov0', 'fov1'],
            channels=['chan0', 'chan1', 'chan2'],
            base_dir=temp_dir,
            cell_cc=cell_cc,
            cell_som_input_data=cluster_data,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating average weighted channel expression files\n"
        assert desired_status_updates in output


def test_generate_remap_avg_wc_files():
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
            'cell_som_cluster': [i for i in np.arange(100)],
            'cell_meta_cluster': [int(i / 5) for i in np.arange(100)],
            'cell_meta_cluster_rename': ['meta' + str(int(i / 5)) for i in np.arange(100)]
        }
        sample_cell_remapping = pd.DataFrame.from_dict(sample_cell_remapping)
        sample_cell_remapping.to_csv(
            os.path.join(temp_dir, 'sample_cell_remapping.csv'),
            index=False
        )

        weighted_channel_comp.generate_remap_avg_wc_files(
            ['fov1', 'fov2'],
            chans,
            temp_dir,
            cluster_data,
            'sample_cell_remapping.csv',
            'sample_weighted_cell_table.feather',
            'sample_cell_som_cluster_chan_avg.csv',
            'sample_cell_meta_cluster_chan_avg.csv'
        )

        # load the re-computed weighted average weighted channel table per cell meta cluster in
        sample_cell_meta_cluster_channel_avg = pd.read_csv(
            os.path.join(temp_dir, 'sample_cell_meta_cluster_chan_avg.csv')
        )

        # assert the markers data has been updated correctly
        result = np.repeat([[0.1, 0.2, 0.3]], repeats=10, axis=0)
        assert np.all(np.round(
            sample_cell_meta_cluster_channel_avg[chans].values, 1) == result
        )

        # assert the correct metacluster labels are contained
        sample_cell_meta_cluster_channel_avg = \
            sample_cell_meta_cluster_channel_avg.sort_values(
                by='cell_meta_cluster'
            )
        assert np.all(sample_cell_meta_cluster_channel_avg[
            'cell_meta_cluster'
        ].values == np.arange(10))
        assert np.all(sample_cell_meta_cluster_channel_avg[
            'cell_meta_cluster_rename'
        ].values == ['meta' + str(i) for i in np.arange(10)])

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
            weighted_channel_comp.generate_weighted_channel_avg_heatmap(
                os.path.join(temp_dir, 'bad_channel_avg.csv'),
                'cell_som_cluster', [], {}, {}
            )

        # basic error check: bad cell cluster col provided
        with pytest.raises(ValueError):
            dummy_chan_avg = pd.DataFrame().to_csv(
                os.path.join(temp_dir, 'sample_channel_avg.csv')
            )
            weighted_channel_comp.generate_weighted_channel_avg_heatmap(
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
            weighted_channel_comp.generate_weighted_channel_avg_heatmap(
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
        weighted_channel_comp.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_som_cluster', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )

        # test 2: cell meta cluster channel avg
        sample_channel_avg = sample_channel_avg.drop(columns='cell_som_cluster')
        sample_channel_avg.to_csv(
            os.path.join(temp_dir, 'sample_channel_avg.csv')
        )

        # assert visualization runs
        weighted_channel_comp.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_meta_cluster_rename', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )
