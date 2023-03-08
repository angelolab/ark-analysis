import os
import tempfile
from shutil import rmtree

import feather
import numpy as np
import pandas as pd
import pytest
from alpineer import io_utils, misc_utils

import ark.phenotyping.cluster_helpers as cluster_helpers
import ark.phenotyping.pixel_som_clustering as pixel_som_clustering

parametrize = pytest.mark.parametrize


def test_run_pixel_som_assignment():
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
                np.random.rand(1000, 4), columns=chans
            )

            # add metadata
            fov_cluster_matrix = pd.concat(
                [fov_cluster_matrix, pd.DataFrame(np.random.rand(1000, 4), columns=meta_colnames)],
                axis=1
            )

            # write the dummy data to pixel_mat_data
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

        # define dummy norm_vals and weights files
        sample_weights = pd.DataFrame(np.random.rand(100, len(chans)), columns=chans)
        sample_norm_vals = pd.DataFrame(np.random.rand(1, len(chans)), columns=chans)
        sample_som_weights_path = os.path.join(temp_dir, 'pixel_weights.feather')
        sample_norm_vals_path = os.path.join(temp_dir, 'norm_vals.feather')
        feather.write_dataframe(sample_weights, sample_som_weights_path)
        feather.write_dataframe(sample_norm_vals, sample_norm_vals_path)

        # define example PixelSOMCluster object
        sample_pixel_cc = cluster_helpers.PixelSOMCluster(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_norm_vals_path,
            sample_som_weights_path, fovs, chans
        )

        fov_status = pixel_som_clustering.run_pixel_som_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov0'
        )

        # assert the fov returned is fov0 and the status is 0
        assert fov_status == ('fov0', 0)

        # read consensus assigned fov0 data in
        consensus_fov_data = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data_temp', 'fov0.feather')
        )

        # assert no SOM cluster assigned a value less than 100
        assert np.all(consensus_fov_data['pixel_som_cluster'].values <= 100)

        # test a corrupted file
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        # attempt to run remapping for fov1
        fov_status = pixel_som_clustering.run_pixel_som_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)


# NOTE: overwrite functionality tested in cluster_helpers_test.py
def test_train_pixel_som():
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.train_pixel_som(
                fovs=['fov0'], channels=['Marker1'],
                base_dir=temp_dir, subset_dir='bad_path'
            )

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

        # create a dummy normalized file
        sample_norm_vals = pd.DataFrame(np.random.rand(1, len(chan_list)), columns=chan_list)
        feather.write_dataframe(
            sample_norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather')
        )

        # not all of the provided fovs exist
        with pytest.raises(ValueError):
            pixel_som_clustering.train_pixel_som(
                fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir
            )

        # column mismatch between provided channels and subsetted data
        with pytest.raises(ValueError):
            pixel_som_clustering.train_pixel_som(
                fovs=fovs, channels=['Marker3', 'Marker4', 'MarkerBad'],
                base_dir=temp_dir
            )

        # train the pixel SOM
        pixel_pysom = pixel_som_clustering.train_pixel_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir
        )

        # assert the weights file has been created
        assert os.path.exists(pixel_pysom.weights_path)

        # assert that the dimensions of the weights are correct
        weights = feather.read_dataframe(pixel_pysom.weights_path)
        assert weights.shape == (100, 4)

        # assert that the SOM weights columns are the same as chan_list
        misc_utils.verify_same_elements(som_weights_channels=weights.columns.values,
                                        provided_channels=chan_list)


def generate_test_pixel_som_cluster_data(temp_dir, fovs, chans,
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

        # write the dummy data to pixel_mat_data
        feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_data',
                                                                 fov + '.feather'))

        fov_data[fov] = fov_cluster_matrix

    # if specified, write fov0 to pixel_mat_data_temp with sample pixel SOM clusters
    if generate_temp:
        # append a dummy meta column to fov0
        fov0_cluster_matrix = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data', 'fov0.feather')
        )
        fov0_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(10), repeats=100)
        feather.write_dataframe(fov0_cluster_matrix, os.path.join(temp_dir,
                                                                  'pixel_mat_data_temp',
                                                                  'fov0.feather'))

    # generate example norm data
    norm_vals_path = os.path.join(temp_dir, 'sample_norm.feather')
    norm_data = pd.DataFrame(np.random.rand(1, len(chans)), columns=chans)
    feather.write_dataframe(norm_data, norm_vals_path)

    # generate example weights
    som_weights_path = os.path.join(temp_dir, 'pixel_weights.feather')
    som_weights_data = pd.DataFrame(np.random.rand(100, len(chans)), columns=chans)
    feather.write_dataframe(som_weights_data, som_weights_path)

    return fov_data, norm_vals_path, som_weights_path


@parametrize('multiprocess', [True, False])
def test_cluster_pixels_base(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        fov_data, norm_vals_path, som_weights_path = generate_test_pixel_som_cluster_data(
            temp_dir, fovs, chan_list
        )

        # error test: weights not assigned to PixelSOMCluster object
        with pytest.raises(ValueError):
            pixel_pysom_bad = cluster_helpers.PixelSOMCluster(
                os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path,
                'bad_path.feather', fovs, chan_list
            )
            pixel_som_clustering.cluster_pixels(fovs, chan_list, temp_dir, pixel_pysom_bad)

        # create a sample PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path, som_weights_path,
            fovs, chan_list
        )

        # run SOM cluster assignment
        pixel_som_clustering.cluster_pixels(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess
        )

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_data',
                                                                   fov + '.feather'))

            # assert we didn't assign any cluster 100 or above
            cluster_ids = fov_cluster_data['pixel_som_cluster']
            assert np.all(cluster_ids <= 100)

        # test overwrite functionality
        capsys.readouterr()

        # run SOM cluster assignment with overwrite flag
        pixel_som_clustering.cluster_pixels(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, reassigning SOM cluster labels to all FOVs\n"
        assert desired_status_updates in output

        # further ensures that all FOVs were overwritten
        assert "There are no more FOVs to assign SOM labels to" not in output


@parametrize('multiprocess', [True, False])
def test_cluster_pixels_corrupt(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        fov_data, norm_vals_path, som_weights_path = generate_test_pixel_som_cluster_data(
            temp_dir, fovs, chans
        )

        # create a sample PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path, som_weights_path, fovs, chans
        )

        # corrupt a fov for this test
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        capsys.readouterr()

        # run SOM cluster assignment
        pixel_som_clustering.cluster_pixels(
            fovs, chans, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess
        )

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


def test_generate_som_avg_files(capsys):
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
            feather.write_dataframe(
                fov_cluster_data, os.path.join(pixel_data_path, fov + '.feather')
            )

        # define a sample norm vals file
        norm_vals_path = os.path.join(temp_dir, 'norm_vals.feather')
        norm_vals = pd.DataFrame(np.random.rand(1, 4), columns=chan_list)
        feather.write_dataframe(norm_vals, norm_vals_path)

        # define a sample weights file
        weights_path = os.path.join(temp_dir, 'pixel_weights.feather')
        weights = pd.DataFrame(np.random.rand(3, 4), columns=chan_list)
        feather.write_dataframe(weights, weights_path)

        # error test: weights not assigned to PixelSOMCluster object
        with pytest.raises(ValueError):
            pixel_pysom_bad = cluster_helpers.PixelSOMCluster(
                pixel_data_path, norm_vals_path, 'bad_path.feather', fovs, chan_list
            )
            pixel_som_clustering.generate_som_avg_files(fovs, chan_list, temp_dir, pixel_pysom_bad)

        # define an example PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            pixel_data_path, norm_vals_path, weights_path, fovs, chan_list
        )

        # test base generation with all subsetted FOVs
        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=3
        )

        # assert we created SOM avg file
        pc_som_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        assert os.path.exists(pc_som_avg_file)

        # load in the SOM avg file, assert all clusters and counts are correct
        # NOTE: more intensive testing done by compute_pixel_cluster_channel_avg
        pc_som_avg_data = pd.read_csv(pc_som_avg_file)
        assert list(pc_som_avg_data['pixel_som_cluster']) == [1, 2, 3]
        assert np.all(pc_som_avg_data['count'] == 100)

        # test that process doesn't run if SOM cluster file already generated
        capsys.readouterr()

        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=1
        )

        output = capsys.readouterr().out
        assert output == "Already generated SOM cluster channel average file, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run SOM averaging with overwrite flg
        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=3,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating SOM cluster channel average file\n"
        assert desired_status_updates in output

        # remove average SOM file for final test
        os.remove(pc_som_avg_file)

        # ensure error gets thrown when not all SOM clusters make it in
        with pytest.raises(ValueError):
            pixel_som_clustering.generate_som_avg_files(
                fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=1
            )
