import os
import tempfile
import warnings
from shutil import rmtree

import feather
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndimage
import skimage.io as io
from skimage.draw import disk
from alpineer import image_utils, misc_utils, test_utils

import ark.phenotyping.cluster_helpers as cluster_helpers
import ark.phenotyping.pixel_cluster_utils as pixel_cluster_utils

parametrize = pytest.mark.parametrize


def test_calculate_channel_percentiles():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), sub_dir="TIFs"
        )

        percentile = 0.5

        # calculate true percentiles
        percentile_dict = {}
        for idx, chan in enumerate(chans):
            chan_vals = []
            for fov in range(len(fovs)):
                current_chan = data_xr[fov, :, :, idx].values
                current_chan = current_chan[current_chan > 0]
                chan_vals.append(np.quantile(current_chan, percentile))

            percentile_dict[chan] = chan_vals

        predicted_percentiles = pixel_cluster_utils.calculate_channel_percentiles(
            tiff_dir=temp_dir,
            channels=chans,
            fovs=fovs,
            img_sub_folder='TIFs',
            percentile=percentile
        )

        # test equality when all channels and all FOVs are included
        for idx, chan in enumerate(chans):
            assert predicted_percentiles[chan].values == np.mean(percentile_dict[chan])

        # include only a subset of channels and fovs
        chans = chans[1:]
        fovs = fovs[:-1]
        predicted_percentiles = pixel_cluster_utils.calculate_channel_percentiles(
            tiff_dir=temp_dir,
            channels=chans,
            fovs=fovs,
            img_sub_folder='TIFs',
            percentile=percentile
        )

        # assert only the specified channels contained in predicted_percentiles
        assert list(predicted_percentiles.columns.values) == chans

        # test equality for specific chans and FOVs
        for idx, chan in enumerate(chans):
            assert predicted_percentiles[chan].values == np.mean(percentile_dict[chan][:-1])


def test_calculate_pixel_intensity_percentile():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(100, 100), sub_dir="TIFs", dtype='float32'
        )

        # make one channel 10x smaller and the other 100x smaller
        for fov in fovs:
            for chan in chans[1:]:
                chan_path = os.path.join(temp_dir, fov, 'TIFs', chan + '.tiff')
                img = io.imread(chan_path)

                if chan == 'chan1':
                    divisor = 10
                else:
                    divisor = 100
                # saved modified channel
                image_utils.save_image(chan_path, img / divisor)

        channel_percentiles = pd.DataFrame(np.array([[1, 1, 1]]),
                                           columns=['chan1', 'chan2', 'chan3'])

        percentile = pixel_cluster_utils.calculate_pixel_intensity_percentile(
            tiff_dir=temp_dir, fovs=fovs, channels=chans,
            img_sub_folder='TIFs', channel_percentiles=channel_percentiles
        )

        assert percentile < 15


# TODO: make the test data more diverse for every function
def test_normalize_rows():
    # define a list of channels and a subset of channels
    chans = ['chan0', 'chan1', 'chan2']
    chan_sub = chans[1:3]

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
    fov_pixel_matrix_sub = pixel_cluster_utils.normalize_rows(fov_pixel_matrix, chan_sub)

    # assert the same channels we subsetted on are found in fov_pixel_matrix_sub
    misc_utils.verify_same_elements(
        provided_chans=chan_sub,
        fov_pixel_chans=fov_pixel_matrix_sub.drop(columns=meta_cols).columns.values
    )

    # assert all the rows sum to 0.5, 0.5
    # this also checks that all the zero-sum rows have been removed
    assert np.all(fov_pixel_matrix_sub.drop(columns=meta_cols).values == [1 / 3, 2 / 3])

    # TEST 2: normalize the matrix and drop the segmentation_label column
    meta_cols.remove('segmentation_label')

    fov_pixel_matrix_sub = pixel_cluster_utils.normalize_rows(
        fov_pixel_matrix, chan_sub, include_seg_label=False
    )

    # assert the same channels we subsetted on are found in fov_pixel_matrix_sub
    misc_utils.verify_same_elements(
        provided_chans=chan_sub,
        fov_pixel_chans=fov_pixel_matrix_sub.drop(columns=meta_cols).columns.values
    )

    # assert all the rows sum to 0.5, 0.5
    # this also checks that all the zero-sum rows have been removed
    assert np.all(fov_pixel_matrix_sub.drop(columns=meta_cols).values == [1 / 3, 2 / 3])


@parametrize('chan_names, err_str', [(['CK18', 'CK17', 'CK18_smoothed'], 'selected CK18'),
                                     (['CK17', 'CK18', 'CK17_nuc_include'], 'selected CK17')])
def test_check_for_modified_channels(chan_names, err_str):
    with tempfile.TemporaryDirectory() as temp_dir:
        test_fov = 'fov1'

        test_fov_path = os.path.join(temp_dir, test_fov)
        os.makedirs(test_fov_path)
        for chan in chan_names:
            test_utils._make_blank_file(test_fov_path, chan + '.tiff')

        selected_chans = chan_names[:-1]

        with pytest.warns(UserWarning, match=err_str):
            pixel_cluster_utils.check_for_modified_channels(
                tiff_dir=temp_dir, test_fov=test_fov, img_sub_folder='', channels=selected_chans
            )

        # check that no warning is raised
        selected_chans = chan_names[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pixel_cluster_utils.check_for_modified_channels(
                tiff_dir=temp_dir, test_fov=test_fov, img_sub_folder='', channels=selected_chans
            )


@parametrize('smooth_vals', [2, [1, 3]])
def test_smooth_channels(smooth_vals):
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), sub_dir="TIFs"
        )
        smooth_channels = ['chan0', 'chan1']

        pixel_cluster_utils.smooth_channels(
            fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
            channels=smooth_channels, smooth_vals=smooth_vals
        )

        # check that correct value was applied
        for fov in fovs:
            for idx, chan in enumerate(chans):

                # get correct image name
                if chan in smooth_channels:
                    suffix = '_smoothed'
                else:
                    suffix = ''
                img = io.imread(os.path.join(temp_dir, fov, 'TIFs', chan + suffix + '.tiff'))

                original_img = data_xr.loc[fov, :, :, chan].values

                # for channels that were smoothed, get correct smooth based on parametrized values
                if chan in smooth_channels:
                    if type(smooth_vals) is int:
                        current_smooth = smooth_vals
                    else:
                        current_smooth = smooth_vals[idx]
                    original_img = ndimage.gaussian_filter(original_img, sigma=current_smooth)

                # check that all channels match expected values
                assert np.array_equal(img, original_img)

        # check that mismatch in list length is caught
        with pytest.raises(ValueError, match='same length'):
            pixel_cluster_utils.smooth_channels(
                fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                channels=smooth_channels, smooth_vals=[1, 2, 3]
            )

        # check that wrong float is caught
        with pytest.raises(ValueError, match='single integer'):
            pixel_cluster_utils.smooth_channels(
                fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                channels=smooth_channels, smooth_vals=1.5
            )

        # check that empty list doesn't raise an error
        pixel_cluster_utils.smooth_channels(
            fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
            channels=[], smooth_vals=smooth_vals
        )


@parametrize('sub_dir', [None, 'TIFs'])
@parametrize('exclude', [False, True])
@parametrize("_nuc_seg_suffix", ["_nuclear.tiff", "_other_suffix.tiff"])
def test_filter_with_nuclear_mask(sub_dir, exclude, _nuc_seg_suffix, capsys):
    # define the fovs to use
    fovs = ['fov0', 'fov1', 'fov2']

    # define the channels to use
    chans = ['chan0', 'chan1']

    with tempfile.TemporaryDirectory() as temp_dir:
        # test seg_dir is None
        pixel_cluster_utils.filter_with_nuclear_mask(
            fovs=fovs, tiff_dir="", seg_dir=None, channel=chans[0], nuc_seg_suffix=_nuc_seg_suffix
        )

        output = capsys.readouterr().out
        assert output == 'No seg_dir provided, you must provide one to run nuclear filtering\n'

        # test invalid seg_dir
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.filter_with_nuclear_mask(
                fovs=fovs, tiff_dir="", seg_dir="bad_seg_path", channel=chans[0],
                nuc_seg_suffix=_nuc_seg_suffix
            )

        # create a directory to store the image data
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)

        # create a segmentation dir
        seg_dir = os.path.join(temp_dir, 'segmentation')
        os.mkdir(seg_dir)

        # define the base ellipse center and radius for the dummy nucleus
        base_center = (4, 4)
        base_radius = 2

        # store the created nuclear centers for future reference
        nuclear_coords = {}

        for offset, fov in enumerate(['fov0', 'fov1', 'fov2']):
            # generate a random segmented image
            rand_img = np.random.randint(1, 16, size=(1, 10, 10))

            # draw a dummy nucleus and store the coords
            nuclear_x, nuclear_y = disk(
                (base_center[0] + offset, base_center[1] + offset), base_radius
            )
            rand_img[0, nuclear_x, nuclear_y] = 0
            nuclear_coords[fov] = (nuclear_x, nuclear_y)

            # save the nuclear segmetation
            file_name = f"{fov}{_nuc_seg_suffix}"
            io.imsave(os.path.join(seg_dir, file_name), rand_img,
                      check_contrast=False)

        # create sample image data
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=fovs,
            channel_names=chans, sub_dir=sub_dir, img_shape=(10, 10)
        )

        # run filtering on channel 0
        pixel_cluster_utils.filter_with_nuclear_mask(
            fovs=fovs, tiff_dir=tiff_dir, seg_dir=seg_dir, channel="chan0",
            nuc_seg_suffix=_nuc_seg_suffix,
            img_sub_folder=sub_dir, exclude=exclude
        )

        # use the correct suffix depending on the exclude arg setting
        suffix = '_nuc_exclude.tiff' if exclude else '_nuc_include.tiff'

        # ensure path correctness if sub_dir is None
        if sub_dir is None:
            sub_dir = ''

        # for each fov, verify that either the nucleus or membrane is all 0
        # depending on exclude arg setting
        for fov in fovs:
            # first assert new channel file was created for channel 0, but not channel 1
            assert os.path.exists(os.path.join(tiff_dir, fov, sub_dir, 'chan0' + suffix))
            assert not os.path.exists(os.path.join(tiff_dir, fov, sub_dir, 'chan1' + suffix))

            # load in the created channel file
            chan_img = io.imread(os.path.join(tiff_dir, fov, sub_dir, 'chan0' + suffix))

            # retrieve the nuclear coords for the fov
            fov_nuclear_x, fov_nuclear_y = nuclear_coords[fov]

            # extract the nuclear and membrane values
            nuclear_vals = chan_img[fov_nuclear_x, fov_nuclear_y]

            mask_arr = np.ones((10, 10), dtype=bool)
            mask_arr[fov_nuclear_x, fov_nuclear_y] = False
            membrane_vals = chan_img[mask_arr]

            # assert the nuclear or membrane channel has been filtered out correctly
            # and the other one is untouched
            if exclude:
                assert not np.all(nuclear_vals == 0)
                assert np.all(membrane_vals == 0)
            else:
                assert np.all(nuclear_vals == 0)
                assert not np.all(membrane_vals == 0)


@parametrize('cluster_col', ['pixel_som_cluster', 'pixel_meta_cluster'])
@parametrize('keep_count', [True, False])
@parametrize('corrupt', [True, False])
def test_compute_pixel_cluster_channel_avg(cluster_col, keep_count, corrupt):
    # define list of fovs and channels
    fovs = ['fov0', 'fov1', 'fov2']
    chans = ['chan0', 'chan1', 'chan2']

    # do not need to test for cluster_dir existence, that happens in consensus_cluster
    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad pixel cluster col passed
        with pytest.raises(ValueError):
            pixel_cluster_utils.compute_pixel_cluster_channel_avg(
                fovs, chans, 'base_dir', 'bad_cluster_col', 100, temp_dir
            )

        # error check: bad num_pixel_clusters provided
        with pytest.raises(ValueError):
            pixel_cluster_utils.compute_pixel_cluster_channel_avg(
                fovs, chans, 'base_dir', 'bad_cluster_col', 0, temp_dir
            )

        # error check: bad num_fovs_subset provided
        with pytest.raises(ValueError):
            pixel_cluster_utils.compute_pixel_cluster_channel_avg(
                fovs, chans, 'base_dir', 'bad_cluster_col', 100, temp_dir, num_fovs_subset=0
            )

        # create a dummy pixel and meta clustered matrix
        # NOTE: while the actual pipeline condenses these into one pixel_mat_data,
        # for this test consider these as the same directory but at different points of the run
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

        # corrupt a file to test functionality
        if corrupt:
            fov0_path = os.path.join(temp_dir, 'pixel_mat_consensus', 'fov0.feather')
            with open(fov0_path, 'w') as outfile:
                outfile.write('baddatabaddatabaddata')

        # define the final result we should get
        if cluster_col == 'pixel_som_cluster':
            num_repeats = 100
            result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)
        else:
            num_repeats = 10
            result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)

        # compute pixel cluster average matrix
        cluster_avg = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, cluster_col,
            100 if cluster_col == 'pixel_som_cluster' else 10,
            'pixel_mat_consensus', num_fovs_subset=1, keep_count=keep_count
        )

        # define the columns to check in cluster_avg
        cluster_avg_cols = chans[:]

        # verify the provided channels and the channels in cluster_avg are exactly the same
        misc_utils.verify_same_elements(
            cluster_avg_chans=cluster_avg[chans].columns.values,
            provided_chans=chans
        )

        # assert count column adds up to just one FOV sampled
        if keep_count:
            assert cluster_avg['count'].sum() == 1000

        # assert all the rows equal [0.1, 0.2, 0.3]
        num_repeats = cluster_avg.shape[0]
        result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)
        assert np.array_equal(result, np.round(cluster_avg[cluster_avg_cols].values, 1))

        # repeat the test but ensure warning for total number of FOVs gets passed properly
        with pytest.warns(UserWarning, match='Provided num_fovs_subset'):
            # compute pixel cluster average matrix
            cluster_avg = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
                fovs[1:], chans, temp_dir, cluster_col,
                100 if cluster_col == 'pixel_som_cluster' else 10,
                'pixel_mat_consensus', num_fovs_subset=3, keep_count=keep_count
            )

            # verify the provided channels and the channels in cluster_avg are exactly the same
            misc_utils.verify_same_elements(
                cluster_avg_chans=cluster_avg[chans].columns.values,
                provided_chans=chans
            )

            # assert count column adds up to just two FOVs sampled
            if keep_count:
                assert cluster_avg['count'].sum() == 2000

            # assert all the rows equal [0.1, 0.2, 0.3]
            num_repeats = cluster_avg.shape[0]
            result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)
            assert np.array_equal(result, np.round(cluster_avg[cluster_avg_cols].values, 1))

        with pytest.raises(ValueError, match='Average expression file not written'):
            pixel_cluster_utils.compute_pixel_cluster_channel_avg(
                fovs[1:], chans, temp_dir, cluster_col, 1000,
                'pixel_mat_consensus', num_fovs_subset=1, keep_count=keep_count
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

        # TEST 1: run fov preprocessing for one fov with seg_labels and no blank pixels
        sample_pixel_mat, sample_pixel_mat_subset = pixel_cluster_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=seg_labels,
            pixel_thresh_val=1
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-4],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-4],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # assert we didn't lose any pixels for this test
        assert sample_pixel_mat.shape[0] == (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TEST 2: run fov preprocessing for one fov without seg_labels and no blank pixels
        sample_pixel_mat, sample_pixel_mat_subset = pixel_cluster_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=None, pixel_thresh_val=1
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-3],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-3],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # assert we didn't lose any pixels for this test
        assert sample_pixel_mat.shape[0] == (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TEST 3: run fov preprocessing with a pixel_thresh_val to ensure rows get removed
        sample_pixel_mat, sample_pixel_mat_subset = pixel_cluster_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data / 1000, seg_labels=seg_labels,
            pixel_thresh_val=0.5
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-4],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-4],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # assert we successfully filtered out pixels below pixel_thresh_val
        assert sample_pixel_mat.shape[0] < (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is less than 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TODO: add a test where after Gaussian blurring one or more rows in sample_pixel_mat
        # are all 0 after, tested successfully via hard-coding values in create_fov_pixel_data


def test_preprocess_fov(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the channel names
        chans = ['chan0', 'chan1', 'chan2']

        # define sample data_dir and subset_dir paths, and make them
        data_dir = os.path.join(temp_dir, 'pixel_mat_data')
        subset_dir = os.path.join(temp_dir, 'pixel_mat_subsetted')
        os.mkdir(data_dir)
        os.mkdir(subset_dir)

        # create sample image data
        # NOTE: test_create_pixel_matrix tests if the sub_dir is None
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=['fov0', 'fov1'],
            channel_names=chans, sub_dir='TIFs', img_shape=(10, 10)
        )

        # create a dummy segmentation directory
        # NOTE: test_create_pixel_matrix handles the case with no segmentation labels provided
        seg_dir = os.path.join(temp_dir, 'segmentation')
        os.mkdir(seg_dir)

        # create sample segmentation data
        for fov in ['fov0', 'fov1']:
            rand_img = np.random.randint(0, 16, size=(10, 10))
            file_name = fov + "_whole_cell.tiff"
            image_utils.save_image(os.path.join(seg_dir, file_name), rand_img)

        channel_norm_df = pd.DataFrame(
            np.expand_dims(np.repeat(10, repeats=len(chans)), axis=0),
            columns=chans
        )

        # run the preprocessing for fov0
        # NOTE: don't test the return value, leave that for test_create_pixel_matrix
        pixel_cluster_utils.preprocess_fov(
            temp_dir, tiff_dir, 'pixel_mat_data', 'pixel_mat_subsetted',
            seg_dir, '_whole_cell.tiff', 'TIFs', False, ['chan0', 'chan1', 'chan2'],
            2, 0.1, 1, 42, channel_norm_df, 'fov0'
        )

        fov_data_path = os.path.join(
            temp_dir, 'pixel_mat_data', 'fov0.feather'
        )
        fov_sub_path = os.path.join(
            temp_dir, 'pixel_mat_subsetted', 'fov0.feather'
        )

        # assert we actually created a .feather preprocessed file
        # for each fov
        assert os.path.exists(fov_data_path)

        # assert that we actually created a .feather subsetted file
        # for each fov
        assert os.path.exists(fov_sub_path)

        # get the data for the specific fov
        flowsom_data_fov = feather.read_dataframe(fov_data_path)
        flowsom_sub_fov = feather.read_dataframe(fov_sub_path)

        # assert the channel names are the same
        misc_utils.verify_same_elements(
            flowsom_chans=flowsom_data_fov.columns.values[:-4],
            provided_chans=chans
        )

        # assert no rows sum to 0
        assert np.all(flowsom_data_fov.loc[:, chans].sum(
            axis=1
        ).values != 0)

        # assert the subsetted DataFrame size is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(flowsom_data_fov.shape[0] * 0.1) == flowsom_sub_fov.shape[0]


def test_find_fovs_missing_col_no_temp():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: provided data path does not exist
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.find_fovs_missing_col(temp_dir, 'bad_data_dir', 'random_col')

        # define a sample data path
        data_path = os.path.join(temp_dir, 'data_dir')
        os.mkdir(data_path)

        # define the FOVs to use
        fovs = ['fov0', 'fov1', 'fov2', 'fov3']

        # write the data for each FOV
        for fov in fovs:
            fov_data = pd.DataFrame(
                np.random.rand(100, 2),
                columns=['chan0', 'chan1']
            )

            feather.write_dataframe(
                fov_data,
                os.path.join(data_path, fov + '.feather')
            )

        # test the case where none of the FOVs have the pixel_som_cluster column
        # all the FOVs should be returned and the _temp dir should be created
        fovs_missing = pixel_cluster_utils.find_fovs_missing_col(
            temp_dir, 'data_dir', 'pixel_som_cluster'
        )
        assert os.path.exists(os.path.join(temp_dir, 'data_dir_temp'))
        misc_utils.verify_same_elements(
            missing_fovs_returned=fovs_missing,
            fovs_without_som=fovs
        )

        # clear data_dir_temp for the next test
        rmtree(os.path.join(temp_dir, 'data_dir_temp'))

        # test the case where all the FOVs have the pixel_som_cluster column
        for fov in fovs:
            fov_data = feather.read_dataframe(os.path.join(temp_dir, 'data_dir', fov + '.feather'))
            fov_data['pixel_som_cluster'] = -1
            feather.write_dataframe(
                fov_data,
                os.path.join(temp_dir, 'data_dir', fov + '.feather')
            )

        # also intentionally corrupt a FOV for the next test
        with open(os.path.join(temp_dir, 'data_dir', 'fov0.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        # no FOVs should be removed, no _temp dir should be created
        # NOTE: handling of corrupted FOVs gets passed to the channel averaging and later steps
        fovs_missing = pixel_cluster_utils.find_fovs_missing_col(
            temp_dir, 'data_dir', 'pixel_som_cluster'
        )
        assert not os.path.exists(os.path.join(temp_dir, 'data_dir_temp'))
        assert fovs_missing == []


def test_find_fovs_missing_col_temp_present():
    with tempfile.TemporaryDirectory() as temp_dir:
        # define a sample data path
        data_path = os.path.join(temp_dir, 'data_dir')
        os.mkdir(data_path)

        # define a sample temp path
        temp_path = os.path.join(temp_dir, 'data_dir_temp')
        os.mkdir(temp_path)

        fovs = ['fov0', 'fov1', 'fov2', 'fov3']
        fovs_som_labels = fovs[:2]

        for fov in fovs:
            fov_data = pd.DataFrame(
                np.random.rand(100, 2),
                columns=['chan0', 'chan1']
            )

            if fov in fovs_som_labels:
                fov_data['pixel_som_cluster'] = -1
                feather.write_dataframe(
                    fov_data,
                    os.path.join(temp_path, fov + '.feather')
                )

            feather.write_dataframe(
                fov_data,
                os.path.join(data_path, fov + '.feather')
            )

        # search for the fovs with the missing pixel_som_cluster column
        fovs_missing = pixel_cluster_utils.find_fovs_missing_col(
            temp_dir, 'data_dir', 'pixel_som_cluster'
        )

        # assert only fov2 and fov3 are contained in the returned list
        misc_utils.verify_same_elements(
            missing_fovs_returned=fovs_missing,
            fovs_without_som=['fov2', 'fov3']
        )


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

        fov_status = pixel_cluster_utils.run_pixel_som_assignment(
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
        fov_status = pixel_cluster_utils.run_pixel_consensus_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)


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

        fov_status = pixel_cluster_utils.run_pixel_consensus_assignment(
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
        fov_status = pixel_cluster_utils.run_pixel_consensus_assignment(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_cc, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)


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
        fov_status = pixel_cluster_utils.update_pixel_meta_labels(
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
        fov_status = pixel_cluster_utils.update_pixel_meta_labels(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_remapped_dict,
            sample_pixel_renamed_meta_dict, 'fov1'
        )

        # assert the fov returned is fov1 and the status is 1
        assert fov_status == ('fov1', 1)
