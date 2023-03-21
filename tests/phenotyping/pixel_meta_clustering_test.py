import os
import tempfile
from copy import deepcopy

import feather
import numpy as np
import pandas as pd
import pytest
from alpineer import io_utils, misc_utils, test_utils

import ark.phenotyping.cluster_helpers as cluster_helpers
import ark.phenotyping.pixel_cluster_utils as pixel_cluster_utils
import ark.phenotyping.pixel_meta_clustering as pixel_meta_clustering

parametrize = pytest.mark.parametrize


def test_run_pixel_consensus_assignment():
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

        # create a dummy data directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

        # create a dummy temp directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        # write dummy clustered data for two fovs
        for fov in ['fov0', 'fov1']:
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

            # assign dummy SOM cluster labels
            fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(1, 101), repeats=10)

            # write the dummy data to pixel_mat_data
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

        # write a sample averaged SOM file per pixel cluster
        sample_pixel_som_avg = pd.DataFrame(
            np.random.rand(100, len(chans)), columns=chans
        )
        sample_pixel_som_avg_path = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        sample_pixel_som_avg.to_csv(sample_pixel_som_avg_path)

        # define example PixelConsensusCluster object
        sample_pixel_cc = cluster_helpers.PixieConsensusCluster(
            'pixel', sample_pixel_som_avg_path, chans
        )

        # force a sample mapping onto sample_pixel_cc
        sample_pixel_cc.mapping = pd.DataFrame.from_dict({
            'pixel_som_cluster': np.arange(1, 101),
            'pixel_meta_cluster': np.repeat(np.arange(1, 21), repeats=5)
        })

        fov_status = pixel_meta_clustering.run_pixel_consensus_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov0'
        )

        # assert the fov returned is fov0 and the status is 0
        assert fov_status == ('fov0', 0)

        # read consensus assigned fov0 data in
        consensus_fov_data = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data_temp', 'fov0.feather')
        )

        # assert the value counts of all renamed meta labels is 50
        assert np.all(consensus_fov_data['pixel_meta_cluster'].value_counts().values == 50)

        # assert each som cluster maps to the right meta cluster
        som_to_meta_gen = consensus_fov_data[
            ['pixel_som_cluster', 'pixel_meta_cluster']
        ].drop_duplicates().sort_values(by='pixel_som_cluster')
        assert np.all(som_to_meta_gen.values == sample_pixel_cc.mapping.values)

        # test a corrupted file
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        # attempt to run remapping for fov1
        fov_status = pixel_meta_clustering.run_pixel_consensus_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)


def generate_test_pixel_consensus_cluster_data(temp_dir, fovs, chans,
                                               generate_temp=False):
    # make it easy to name metadata columns
    meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

    # create a dummy clustered matrix
    os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

    # create a dummy temp directory if specified
    if generate_temp:
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data_temp'))

    # store the intermediate FOV data in a dict for future comparison
    fov_data = {}

    # write dummy clustered data for each fov
    for fov in fovs:
        # create dummy preprocessed data for each fov
        fov_cluster_matrix = pd.DataFrame(
            np.random.rand(1000, len(chans)),
            columns=chans
        )

        # assign dummy metadata labels
        fov_cluster_matrix['fov'] = fov
        fov_cluster_matrix['row_index'] = np.repeat(np.arange(1, 101), repeats=10)
        fov_cluster_matrix['column_index'] = np.tile(np.arange(1, 101), reps=10)
        fov_cluster_matrix['segmentation_label'] = np.arange(1, 1001)

        # assign dummy cluster labels
        fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(100), repeats=10)

        # write the dummy data to pixel_mat_data
        feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_data',
                                                                 fov + '.feather'))

        fov_data[fov] = fov_cluster_matrix

    # if specified, write fov0 to pixel_mat_data_temp with sample pixel meta clusters
    if generate_temp:
        # append a dummy meta column to fov0
        fov0_cluster_matrix = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data', 'fov0.feather')
        )
        fov0_cluster_matrix['pixel_meta_cluster'] = np.repeat(np.arange(10), repeats=100)
        feather.write_dataframe(fov0_cluster_matrix, os.path.join(temp_dir,
                                                                  'pixel_mat_data_temp',
                                                                  'fov0.feather'))

    # compute averages by SOM cluster
    cluster_avg = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
        fovs, chans, temp_dir, 'pixel_som_cluster', 100
    )

    # save the DataFrame
    cluster_avg.to_csv(
        os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv'),
        index=False
    )

    return fov_data


@parametrize('multiprocess', [True, False])
def test_pixel_consensus_cluster_base(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        fov_data = generate_test_pixel_consensus_cluster_data(
            temp_dir, fovs, chans
        )

        # run consensus clustering
        pixel_cc = pixel_meta_clustering.pixel_consensus_cluster(
            fovs=fovs, channels=chans, base_dir=temp_dir
        )

        # assert we assigned a mapping, then sort
        assert pixel_cc.mapping is not None
        sample_mapping = deepcopy(pixel_cc.mapping)
        sample_mapping = sample_mapping.sort_values(by='pixel_som_cluster')

        for fov in fovs:
            fov_consensus_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

            # assert all assigned SOM cluster values contained in original fov-data
            # NOTE: can't test exact values because of randomization of channel averaging
            misc_utils.verify_in_list(
                assigned_som_values=fov_consensus_data['pixel_som_cluster'].unique(),
                valid_som_values=fov_data[fov]['pixel_som_cluster']
            )

            # assert we didn't assign any cluster 20 or above
            consensus_cluster_ids = fov_consensus_data['pixel_meta_cluster']
            assert np.all(consensus_cluster_ids <= 20)

            # assert the correct labels have been assigned
            fov_mapping = fov_consensus_data[
                ['pixel_som_cluster', 'pixel_meta_cluster']
            ].drop_duplicates().sort_values(by='pixel_som_cluster')

            assert np.all(sample_mapping.values == fov_mapping.values)

        # run consensus clustering with the overwrite flag
        pixel_cc = pixel_meta_clustering.pixel_consensus_cluster(
            fovs=fovs, channels=chans, base_dir=temp_dir, overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, reassigning meta cluster labels to all FOVs\n"
        assert desired_status_updates in output

        # further ensures that all FOVs were overwritten
        assert "There are no more FOVs to assign meta labels to" not in output


@parametrize('multiprocess', [True, False])
def test_pixel_consensus_cluster_corrupt(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        generate_test_pixel_consensus_cluster_data(
            temp_dir, fovs, chans, generate_temp=True
        )

        # corrupt a fov for this test
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        capsys.readouterr()

        # run the consensus clustering process
        pixel_meta_clustering.pixel_consensus_cluster(fovs=fovs, channels=chans, base_dir=temp_dir)

        # assert the _temp folder is now gone
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        output = capsys.readouterr().out
        desired_status_updates = "The data for FOV fov1 has been corrupted, skipping\n"
        assert desired_status_updates in output

        # verify that the FOVs in pixel_mat_data are correct
        # NOTE: fov1 should not be written because it was corrupted
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=['fov0.feather', 'fov2.feather']
        )


def test_generate_meta_avg_files(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

        # define sample pixel data for each FOV
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_dir')
        os.mkdir(pixel_data_path)
        for i, fov in enumerate(fovs):
            fov_cluster_data = pd.DataFrame(np.random.rand(100, len(colnames)), columns=colnames)
            fov_cluster_data['pixel_som_cluster'] = i + 1
            fov_cluster_data['pixel_meta_cluster'] = (i + 1) * 10
            feather.write_dataframe(
                fov_cluster_data, os.path.join(pixel_data_path, fov + '.feather')
            )

        # define a sample pixel channel SOM average file
        pc_som_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        pc_som_avg_data = pd.DataFrame(
            np.random.rand(3, len(chan_list) + 2),
            columns=['pixel_som_cluster'] + chan_list + ['count']
        )
        pc_som_avg_data['pixel_som_cluster'] = np.arange(1, 4)
        pc_som_avg_data['count'] = 100
        pc_som_avg_data.to_csv(pc_som_avg_file, index=False)

        # define a sample SOM to meta cluster map
        som_to_meta_data = {
            'pixel_som_cluster': np.arange(1, 4, dtype=np.int64),
            'pixel_meta_cluster': np.arange(10, 40, 10, dtype=np.int64)
        }
        som_to_meta_data = pd.DataFrame.from_dict(som_to_meta_data)

        # define a sample ConsensusCluster object
        # define a dummy input file for data, we won't need it for expression average testing
        consensus_dummy_file = os.path.join(temp_dir, 'dummy_consensus_input.csv')
        pd.DataFrame().to_csv(consensus_dummy_file)
        pixel_cc = cluster_helpers.PixieConsensusCluster(
            'pixel', consensus_dummy_file, chan_list, max_k=3
        )
        pixel_cc.mapping = som_to_meta_data

        # test base generation with all subsetted FOVs
        pixel_meta_clustering.generate_meta_avg_files(
            fovs, chan_list, temp_dir, pixel_cc, 'pixel_data_dir', num_fovs_subset=3
        )

        # assert we created meta avg file
        pc_meta_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_meta_cluster.csv')
        assert os.path.exists(pc_som_avg_file)

        # load in the meta avg file, assert all clusters and counts are correct
        # NOTE: more intensive testing done by compute_pixel_cluster_channel_avg
        pc_meta_avg_data = pd.read_csv(pc_meta_avg_file)
        assert list(pc_meta_avg_data['pixel_meta_cluster']) == [10, 20, 30]
        assert np.all(pc_som_avg_data['count'] == 100)

        # ensure that the mapping in the SOM average file is correct
        pc_som_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        pc_som_avg_data = pd.read_csv(pc_som_avg_file)
        pc_som_mapping = pc_som_avg_data[['pixel_som_cluster', 'pixel_meta_cluster']]
        assert np.all(pc_som_mapping.values == som_to_meta_data.values)

        # test that process doesn't run if meta cluster file already generated
        capsys.readouterr()

        pixel_meta_clustering.generate_meta_avg_files(
            fovs, chan_list, temp_dir, pixel_cc, 'pixel_data_dir', num_fovs_subset=1
        )

        output = capsys.readouterr().out
        assert output == "Already generated meta cluster channel average file, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run SOM averaging with overwrite flg
        pixel_meta_clustering.generate_meta_avg_files(
            fovs, chan_list, temp_dir, pixel_cc, 'pixel_data_dir', num_fovs_subset=3,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating meta cluster channel average file\n"
        assert desired_status_updates in output

        # remove average meta file for final test
        os.remove(pc_meta_avg_file)

        # ensure error gets thrown when not all meta clusters make it in
        with pytest.raises(ValueError):
            pixel_meta_clustering.generate_meta_avg_files(
                fovs, chan_list, temp_dir, pixel_cc, 'pixel_data_dir', num_fovs_subset=1
            )


def test_update_pixel_meta_labels():
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

        # create a dummy data directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

        # create a dummy temp directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        # write dummy clustered data for two fovs
        for fov in ['fov0', 'fov1']:
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

            # assign dummy SOM cluster labels
            fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(100), repeats=10)

            # assign dummy meta cluster labels
            fov_cluster_matrix['pixel_meta_cluster'] = np.repeat(np.arange(10), repeats=100)

            # write the dummy data to pixel_mat_data
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

        # define dummy remap schemes
        sample_pixel_remapped_dict = {i: int(i / 5) for i in np.arange(100)}
        sample_pixel_renamed_meta_dict = {i: 'meta_' + str(i) for i in sample_pixel_remapped_dict}

        # run remapping for fov0
        fov_status = pixel_meta_clustering.update_pixel_meta_labels(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_remapped_dict,
            sample_pixel_renamed_meta_dict, 'fov0'
        )

        # assert the fov returned is fov0 and the status is 0
        assert fov_status == ('fov0', 0)

        # read remapped fov0 data in
        remapped_fov_data = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data_temp', 'fov0.feather')
        )

        # assert the value counts of all renamed meta labels is 50
        assert np.all(remapped_fov_data['pixel_meta_cluster_rename'].value_counts().values == 50)

        # assert each meta cluster label maps to the right renamed cluster
        remapped_meta_info = dict(
            remapped_fov_data[
                ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
            ].drop_duplicates().values
        )
        for meta_cluster in remapped_meta_info:
            assert remapped_meta_info[meta_cluster] == sample_pixel_renamed_meta_dict[meta_cluster]

        # test a corrupted file
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        # attempt to run remapping for fov1
        fov_status = pixel_meta_clustering.update_pixel_meta_labels(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_remapped_dict,
            sample_pixel_renamed_meta_dict, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)


def generate_test_apply_pixel_meta_cluster_remapping_data(temp_dir, fovs, chans,
                                                          generate_temp=False):
    # make it easy to name metadata columns
    meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

    # create a dummy data directory
    os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

    # create a dummy temp directory if specified
    if generate_temp:
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data_temp'))

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

        # assign dummy SOM cluster labels
        fov_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(100), repeats=10)

        # assign dummy meta cluster labels
        fov_cluster_matrix['pixel_meta_cluster'] = np.repeat(np.arange(10), repeats=100)

        # write the dummy data to pixel_mat_data
        feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_data',
                                                                 fov + '.feather'))

    # if specified, write write fov0 to pixel_mat_data_temp with sample renamed pixel meta clusters
    if generate_temp and fov == 'fov0':
        # append a dummy meta column to fov0
        fov0_cluster_matrix = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data', 'fov0.feather')
        )
        fov0_cluster_matrix['pixel_meta_cluster_rename'] = np.repeat(np.arange(10), repeats=100)
        feather.write_dataframe(fov0_cluster_matrix, os.path.join(temp_dir,
                                                                  'pixel_mat_data_temp',
                                                                  'fov0.feather'))

    # define a dummy remap scheme and save
    # NOTE: we intentionally add more SOM cluster keys than necessary to show
    # that certain FOVs don't need to contain every SOM cluster available
    sample_pixel_remapping = {
        'pixel_som_cluster': [i for i in np.arange(105)],
        'pixel_meta_cluster': [int(i / 5) for i in np.arange(105)],
        'pixel_meta_cluster_rename': ['meta' + str(int(i / 5)) for i in np.arange(105)]
    }
    sample_pixel_remapping = pd.DataFrame.from_dict(sample_pixel_remapping)
    sample_pixel_remapping.to_csv(
        os.path.join(temp_dir, 'sample_pixel_remapping.csv'),
        index=False
    )

    # make a basic average channel per SOM cluster file
    pixel_som_cluster_channel_avgs = pd.DataFrame(
        np.repeat(np.array([[0.1, 0.2, 0.3, 0.4]]), repeats=100, axis=0)
    )
    pixel_som_cluster_channel_avgs['pixel_som_cluster'] = np.arange(100)
    pixel_som_cluster_channel_avgs['pixel_meta_cluster'] = np.repeat(
        np.arange(10), repeats=10
    )
    pixel_som_cluster_channel_avgs.to_csv(
        os.path.join(temp_dir, 'sample_pixel_som_cluster_chan_avgs.csv'), index=False
    )

    # since the average channel per meta cluster file will be completely overwritten,
    # just make it a blank slate
    pd.DataFrame().to_csv(
        os.path.join(temp_dir, 'sample_pixel_meta_cluster_chan_avgs.csv'), index=False
    )


# TODO: split up this test function
@parametrize('multiprocess', [True, False])
def test_apply_pixel_meta_cluster_remapping_base(multiprocess):
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad path to pixel consensus dir
        with pytest.raises(FileNotFoundError):
            pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'bad_consensus_dir',
                'remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy consensus dir
        os.mkdir(os.path.join(temp_dir, 'pixel_consensus_dir'))

        # basic error check: bad path to remapped name
        with pytest.raises(FileNotFoundError):
            pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'bad_remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # generate the test environment
        generate_test_apply_pixel_meta_cluster_remapping_data(temp_dir, fovs, chans)

        # error check: bad columns provided in the SOM to meta cluster map csv input
        with pytest.raises(ValueError):
            sample_pixel_remapping = pd.read_csv(
                os.path.join(temp_dir, 'sample_pixel_remapping.csv')
            )
            bad_sample_pixel_remapping = sample_pixel_remapping.copy()
            bad_sample_pixel_remapping = bad_sample_pixel_remapping.rename(
                {'pixel_meta_cluster_rename': 'bad_col'},
                axis=1
            )
            bad_sample_pixel_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_pixel_remapping.csv'),
                index=False
            )

            pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
                fovs,
                chans,
                temp_dir,
                'pixel_mat_data',
                'bad_sample_pixel_remapping.csv'
            )

        # error check: mapping does not contain every SOM label
        with pytest.raises(ValueError):
            bad_sample_pixel_remapping = {
                'pixel_som_cluster': [1, 2],
                'pixel_meta_cluster': [1, 2],
                'pixel_meta_cluster_rename': ['m1', 'm2']
            }
            bad_sample_pixel_remapping = pd.DataFrame.from_dict(bad_sample_pixel_remapping)
            bad_sample_pixel_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_pixel_remapping.csv'),
                index=False
            )

            pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
                fovs,
                chans,
                temp_dir,
                'pixel_mat_data',
                'bad_sample_pixel_remapping.csv'
            )

        # run the remapping process
        pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
            fovs,
            chans,
            temp_dir,
            'pixel_mat_data',
            'sample_pixel_remapping.csv',
            multiprocess=multiprocess
        )

        # assert _temp dir no longer exists (pixel_mat_data_temp should be renamed pixel_mat_data)
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        # used for mapping verification
        actual_som_to_meta = sample_pixel_remapping[
            ['pixel_som_cluster', 'pixel_meta_cluster']
        ].drop_duplicates().sort_values(by='pixel_som_cluster')
        actual_meta_id_to_name = sample_pixel_remapping[
            ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ].drop_duplicates().sort_values(by='pixel_meta_cluster')

        for fov in fovs:
            # read remapped fov data in
            remapped_fov_data = feather.read_dataframe(
                os.path.join(temp_dir, 'pixel_mat_data', fov + '.feather')
            )

            # assert the counts for each FOV on every meta cluster is 50
            assert np.all(remapped_fov_data['pixel_meta_cluster'].value_counts().values == 50)

            # assert the mapping is the same for pixel SOM to meta cluster
            som_to_meta = remapped_fov_data[
                ['pixel_som_cluster', 'pixel_meta_cluster']
            ].drop_duplicates().sort_values(by='pixel_som_cluster')

            # this tests the case where a FOV doesn't necessarily need to have all the possible
            # SOM clusters in it
            actual_som_to_meta_subset = actual_som_to_meta[
                actual_som_to_meta['pixel_som_cluster'].isin(
                    som_to_meta['pixel_som_cluster']
                )
            ]

            assert np.all(som_to_meta.values == actual_som_to_meta_subset.values)

            # assert the mapping is the same for pixel meta cluster to renamed pixel meta cluster
            meta_id_to_name = remapped_fov_data[
                ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
            ].drop_duplicates().sort_values(by='pixel_meta_cluster')

            # this tests the case where a FOV doesn't necessarily need to have all the possible
            # meta clusters in it
            actual_meta_id_to_name_subset = actual_meta_id_to_name[
                actual_meta_id_to_name['pixel_meta_cluster'].isin(
                    meta_id_to_name['pixel_meta_cluster']
                )
            ]

            assert np.all(meta_id_to_name.values == actual_meta_id_to_name_subset.values)


@parametrize('multiprocess', [True, False])
def test_apply_pixel_meta_cluster_remapping_temp_corrupt(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # generate the test environment
        generate_test_apply_pixel_meta_cluster_remapping_data(
            temp_dir, fovs, chans, generate_temp=True
        )

        # corrupt a fov for this test
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        capsys.readouterr()

        # run the remapping process
        pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
            fovs,
            chans,
            temp_dir,
            'pixel_mat_data',
            'sample_pixel_remapping.csv',
            multiprocess=multiprocess
        )

        # assert the _temp folder is now gone
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        output = capsys.readouterr().out
        desired_status_updates = "Using re-mapping scheme to re-label pixel meta clusters\n"
        desired_status_updates += "The data for FOV fov1 has been corrupted, skipping\n"
        assert desired_status_updates in output

        # verify that the FOVs in pixel_mat_data are correct
        # NOTE: fov1 should not be written because it was corrupted
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=['fov0.feather', 'fov2.feather']
        )


def test_generate_remap_avg_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

        # define sample pixel data for each FOV
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_dir')
        os.mkdir(pixel_data_path)
        for i, fov in enumerate(fovs):
            fov_cluster_data = pd.DataFrame(np.random.rand(100, len(colnames)), columns=colnames)
            fov_cluster_data['pixel_som_cluster'] = i + 1
            fov_cluster_data['pixel_meta_cluster'] = (i + 1) * 10
            feather.write_dataframe(
                fov_cluster_data, os.path.join(pixel_data_path, fov + '.feather')
            )

        # define a sample pixel channel SOM average file
        pc_som_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        pc_som_avg_data = pd.DataFrame(
            np.random.rand(3, len(chan_list) + 2),
            columns=['pixel_som_cluster'] + chan_list + ['count']
        )
        pc_som_avg_data['pixel_som_cluster'] = np.arange(1, 4)
        pc_som_avg_data['count'] = 100
        pc_som_avg_data.to_csv(pc_som_avg_file, index=False)

        # define a sample pixel channel meta average file
        pc_meta_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_meta_cluster.csv')
        pc_meta_avg_data = pd.DataFrame(
            np.random.rand(3, len(chan_list) + 3),
            columns=['pixel_som_cluster', 'pixel_meta_cluster'] + chan_list + ['count']
        )
        pc_meta_avg_data['pixel_som_cluster'] = np.arange(1, 4)
        pc_meta_avg_data['pixel_meta_cluster'] = np.arange(10, 40, 10)
        pc_meta_avg_data['count'] = 100
        pc_meta_avg_data.to_csv(pc_meta_avg_file, index=False)

        # define a sample meta remap file
        meta_remap_path = os.path.join(temp_dir, 'meta_remap.csv')
        meta_remap_data = {
            'pixel_som_cluster': np.arange(1, 4),
            'pixel_meta_cluster': np.arange(10, 40, 10),
            'pixel_meta_cluster_rename': [f'meta_rename_{i}' for i in np.arange(10, 40, 10)]
        }
        meta_remap_data = pd.DataFrame.from_dict(meta_remap_data)
        meta_remap_data.to_csv(meta_remap_path, index=False)

        # test base generation with all subsetted FOVs
        pixel_meta_clustering.generate_remap_avg_files(
            fovs, chan_list, temp_dir, 'pixel_data_dir', 'meta_remap.csv',
            'pixel_channel_avg_som_cluster.csv', 'pixel_channel_avg_meta_cluster.csv',
            num_fovs_subset=3
        )

        # load in the meta average file and assert the mappings are correct
        pc_meta_remap_data = pd.read_csv(pc_meta_avg_file)
        pc_meta_mappings = pc_meta_remap_data[
            ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ]
        assert np.all(
            pc_meta_mappings.values == meta_remap_data.drop(columns='pixel_som_cluster').values
        )

        # load in the som average file and assert the mappings are correct
        pc_som_remap_data = pd.read_csv(pc_som_avg_file)
        pc_som_mappings = pc_som_remap_data[
            ['pixel_som_cluster', 'pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ]
        assert np.all(
            pc_som_mappings.values == meta_remap_data.values
        )

        # ensure error gets thrown when not all meta clusters make it in
        with pytest.raises(ValueError):
            pixel_meta_clustering.generate_remap_avg_files(
                fovs, chan_list, temp_dir, 'pixel_data_dir', 'meta_remap.csv',
                'pixel_channel_avg_som_cluster.csv', 'pixel_channel_avg_meta_cluster.csv',
                num_fovs_subset=1
            )
