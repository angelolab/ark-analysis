import json
import os
import pytest
from pytest_cases import parametrize_with_cases
import tempfile
import warnings

import feather
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import skimage.io as io
import scipy.ndimage as ndimage
from sklearn.utils import shuffle
import xarray as xr

import ark.phenotyping.som_utils as som_utils
import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

parametrize = pytest.mark.parametrize


PIXEL_MATRIX_FOVS = ['fov0', 'fov1', 'fov2']
PIXEL_MATRIX_CHANS = ['chan0', 'chan1', 'chan2']


class CreatePixelMatrixCases:
    def case_all_fovs_all_chans(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, False

    def case_some_fovs_some_chans(self):
        return PIXEL_MATRIX_FOVS[:2], PIXEL_MATRIX_CHANS[:2], 'TIFs', True, False, False

    def case_no_sub_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, None, True, False, False

    def case_no_seg_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', False, False, False

    def case_existing_channel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, False

    def case_existing_pixel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, True


def mocked_train_pixel_som(fovs, channels, base_dir,
                           subset_dir='pixel_mat_subsetted',
                           norm_vals_name='post_rowsum_chan_norm.feather',
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


def mocked_cluster_pixels(fovs, channels, base_dir, data_dir='pixel_mat_data',
                          norm_vals_name='post_rowsum_chan_norm.feather',
                          weights_name='pixel_weights.feather',
                          pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                          batch_size=5):
    # read in the norm_vals matrix
    norm_vals = feather.read_dataframe(os.path.join(base_dir, norm_vals_name))

    # read in the weights matrix
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))

    for fov in fovs:
        # read the specific fov from the preprocessed feather
        fov_mat_pre = feather.read_dataframe(os.path.join(base_dir, data_dir, fov + '.feather'))

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
                                                          data_dir,
                                                          fov + '.feather'))


def mocked_pixel_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                                   data_dir='pixel_mat_data',
                                   pc_chan_avg_som_cluster_name='pixel_chan_avg_som_cluster.csv',
                                   pc_chan_avg_meta_cluster_name='pixel_chan_avg_meta_cluster.csv',
                                   clust_to_meta_name='pixel_clust_to_meta.feather',
                                   batch_size=5, seed=42):
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
                                                                 data_dir,
                                                                 fov + '.feather'))

        # use mapping to assign hierarchical cluster ids
        fov_cluster_matrix = pd.merge(fov_cluster_matrix, hClust_to_clust)

        # write consensus cluster results to feather
        feather.write_dataframe(fov_cluster_matrix, os.path.join(base_dir,
                                                                 data_dir,
                                                                 fov + '.feather'))


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


def mocked_create_fov_pixel_data(fov, channels, img_data, seg_labels, blur_factor,
                                 subset_proportion, pixel_norm_val):
    print("CALLING MOCKED CREATE FOV PIXEL DATA")
    # create fake data to be compatible with downstream functions
    data = np.random.rand(len(channels) * 5).reshape(5, len(channels))
    df = pd.DataFrame(data, columns=channels)

    # verify that each channel is 2x the previous
    for i in range(len(channels) - 1):
        assert np.allclose(img_data[..., i] * 2, img_data[..., i + 1])

    return df, df


def mocked_preprocess_fov(base_dir, tiff_dir, data_dir, subset_dir, seg_dir, seg_suffix,
                          img_sub_folder, is_mibitiff, channels, blur_factor,
                          subset_proportion, pixel_norm_val, dtype, fov):
    # load img_xr from MIBITiff or directory with the fov
    if is_mibitiff:
        img_data = load_utils.load_imgs_from_mibitiff(
            tiff_dir, mibitiff_files=[fov], dtype=dtype
        )
    else:
        img_data = load_utils.load_imgs_from_tree(
            tiff_dir, img_sub_folder=img_sub_folder, fovs=[fov], dtype=dtype
        )

    mocked_create_fov_pixel_data(base_dir, channels, img_data.values, np.random.rand(),
                                 blur_factor, subset_proportion, np.random.rand())


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

        predicted_percentiles = som_utils.calculate_channel_percentiles(tiff_dir=temp_dir,
                                                                        channels=chans,
                                                                        fovs=fovs,
                                                                        img_sub_folder='TIFs',
                                                                        percentile=percentile)
        # test equality when all channels and all FOVs are included
        for idx, chan in enumerate(chans):
            assert predicted_percentiles['norm_val'].values[idx] == np.mean(percentile_dict[chan])

        # include only a subset of channels and fovs
        chans = chans[1:]
        fovs = fovs[:-1]
        predicted_percentiles = som_utils.calculate_channel_percentiles(tiff_dir=temp_dir,
                                                                        channels=chans,
                                                                        fovs=fovs,
                                                                        img_sub_folder='TIFs',
                                                                        percentile=percentile)
        # test equality for specific chans and FOVs
        for idx, chan in enumerate(chans):
            assert predicted_percentiles['norm_val'].values[idx] == \
                   np.mean(percentile_dict[chan][:-1])


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
                io.imsave(chan_path, img / divisor)

        channel_percentiles = pd.DataFrame({'channel': ['chan1', 'chan2', 'chan3'],
                                            'norm_val': [1, 1, 1]})
        percentile = som_utils.calculate_pixel_intensity_percentile(
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
    fov_pixel_matrix_sub = som_utils.normalize_rows(fov_pixel_matrix, chan_sub)

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
            som_utils.check_for_modified_channels(tiff_dir=temp_dir, test_fov=test_fov,
                                                  img_sub_folder='', channels=selected_chans)

        # check that no warning is raised
        selected_chans = chan_names[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            som_utils.check_for_modified_channels(tiff_dir=temp_dir, test_fov=test_fov,
                                                  img_sub_folder='', channels=selected_chans)


@parametrize('smooth_vals', [2, [1, 3]])
def test_smooth_channels(smooth_vals):
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), sub_dir="TIFs"
        )
        smooth_channels = ['chan0', 'chan1']

        som_utils.smooth_channels(fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                                  channels=smooth_channels, smooth_vals=smooth_vals)

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
            som_utils.smooth_channels(fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                                      channels=smooth_channels, smooth_vals=[1, 2, 3])

        # check that wrong float is caught
        with pytest.raises(ValueError, match='single integer'):
            som_utils.smooth_channels(fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                                      channels=smooth_channels, smooth_vals=1.5)

        # check that empty list doesn't raise an error
        som_utils.smooth_channels(fovs=fovs, tiff_dir=temp_dir, img_sub_folder='TIFs',
                                  channels=[], smooth_vals=smooth_vals)


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

        for cluster_col in ['pixel_som_cluster', 'pixel_meta_cluster']:
            # define the final result we should get
            if cluster_col == 'pixel_som_cluster':
                num_repeats = 100
                result = np.repeat(np.array([[0.1, 0.2, 0.3]]), repeats=num_repeats, axis=0)
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


def test_compute_cell_cluster_count_avg():
    # define the cluster columns
    pixel_som_clusters = ['pixel_som_cluster_%d' % i for i in np.arange(3)]
    pixel_meta_clusters = ['pixel_meta_cluster_rename_%s' % str(i) for i in np.arange(3)]

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

                cell_cluster_avg = som_utils.compute_cell_cluster_count_avg(
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

                cell_cluster_avg = som_utils.compute_cell_cluster_count_avg(
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
        weighted_cell_table.to_csv(
            os.path.join(temp_dir, 'weighted_cell_channel.csv'),
            index=False
        )

        # error check: bad cell_cluster_col provided
        with pytest.raises(ValueError):
            som_utils.compute_cell_cluster_channel_avg(
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
        cell_channel_avg = som_utils.compute_cell_cluster_channel_avg(
            # fovs, chans, temp_dir, weighted_cell_table,
            fovs, chans, temp_dir, 'weighted_cell_channel.csv',
            'cell_mat_consensus.feather', cell_cluster_col='cell_som_cluster'
        )

        # assert the same SOM clusters were assigned
        assert np.all(cell_channel_avg['cell_som_cluster'].values == np.arange(5))

        # assert the returned shape is correct
        assert cell_channel_avg.drop(columns='cell_som_cluster').shape == (5, 3)

        # test averages for cell meta clusters
        cell_channel_avg = som_utils.compute_cell_cluster_channel_avg(
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
            cell_counts, _ = som_utils.create_c2pc_data(
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
                som_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, fovs=['fov2', 'fov3']
                )

            # error check: invalid pixel_cluster_col provided
            with pytest.raises(ValueError):
                som_utils.compute_p2c_weighted_channel_avg(
                    cluster_avg, chans, cell_counts, pixel_cluster_col='bad_cluster_col'
                )

            # test for all and some fovs
            for fov_list in [None, fovs[:1]]:
                # test with som cluster counts and all fovs
                channel_avg = som_utils.compute_p2c_weighted_channel_avg(
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
            som_utils.create_c2pc_data(
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

            cluster_counts, cluster_counts_norm = som_utils.create_c2pc_data(
                fovs, pixel_data_path, bad_cell_table_path,
                pixel_cluster_col='pixel_som_cluster'
            )

        # test counts on the pixel cluster column
        cluster_counts, cluster_counts_norm = som_utils.create_c2pc_data(
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
        cluster_counts, cluster_counts_norm = som_utils.create_c2pc_data(
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
        sample_pixel_mat, sample_pixel_mat_subset = som_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=seg_labels,
            pixel_norm_val=1
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
        sample_pixel_mat, sample_pixel_mat_subset = som_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=None, pixel_norm_val=1
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

        # TEST 3: run fov preprocessing with a pixel_norm_val to ensure rows get removed
        sample_pixel_mat, sample_pixel_mat_subset = som_utils.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data / 1000, seg_labels=seg_labels,
            pixel_norm_val=0.5
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-4],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-4],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # for this test, we want to ensure we successfully filtered out pixels below pixel_norm_val
        assert sample_pixel_mat.shape[0] < (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is less than 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TODO: add a test where after Gaussian blurring one or more rows in sample_pixel_mat
        # are all 0 after, tested successfully via hard-coding values in create_fov_pixel_data


def test_preprocess_fov():
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
            file_name = fov + "_feature_0.tif"
            io.imsave(os.path.join(seg_dir, file_name), rand_img,
                      check_contrast=False)

        # run the preprocessing for fov0
        # NOTE: don't test the return value, leave that for test_create_pixel_matrix
        som_utils.preprocess_fov(
            temp_dir, tiff_dir, 'pixel_mat_data', 'pixel_mat_subsetted',
            seg_dir, '_feature_0.tif', 'TIFs', False, ['chan0', 'chan1', 'chan2'],
            2, 0.1, 1, 'int16', 42, 'fov0'
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


# TODO: leaving out MIBItiff testing until someone needs it
@parametrize_with_cases(
    'fovs,chans,sub_dir,seg_dir_include,channel_norm_include,pixel_norm_include',
    cases=CreatePixelMatrixCases
)
def test_create_pixel_matrix(fovs, chans, sub_dir, seg_dir_include,
                             channel_norm_include, pixel_norm_include, mocker):
    # sample_labels = test_utils.make_labels_xarray(label_data=None,
    #                                               fov_ids=['fov0', 'fov1', 'fov2'],
    #                                               compartment_names=['whole_cell'])

    with tempfile.TemporaryDirectory() as temp_dir:
        # create a directory to store the image data
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)

        # invalid subset proportion specified
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          channels=['chan1'],
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          seg_dir=None,
                                          subset_proportion=1.1)

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          channels=['chan1'],
                                          base_dir='bad_base_dir',
                                          tiff_dir=tiff_dir,
                                          seg_dir=None)

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2'],
                                          channels=['chan1'],
                                          base_dir=temp_dir,
                                          tiff_dir='bad_tiff_dir',
                                          seg_dir=None)

        # create a dummy seg_dir with data if we're on a test that requires segmentation labels
        if seg_dir_include:
            seg_dir = os.path.join(temp_dir, 'segmentation')
            os.mkdir(seg_dir)

            # create sample segmentation data
            for fov in fovs:
                rand_img = np.random.randint(0, 16, size=(10, 10))
                file_name = fov + "_feature_0.tif"
                io.imsave(os.path.join(seg_dir, file_name), rand_img,
                          check_contrast=False)
        # otherwise, set seg_dir to None
        else:
            seg_dir = None

        # create sample image data
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=fovs,
            channel_names=['chan0', 'chan1', 'chan2'], sub_dir=sub_dir, img_shape=(10, 10)
        )

        # pass invalid fov names (fails in load_imgs_from_tree)
        with pytest.raises(FileNotFoundError):
            som_utils.create_pixel_matrix(fovs=['fov1', 'fov2', 'fov3'],
                                          channels=chans,
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          img_sub_folder=sub_dir,
                                          seg_dir=seg_dir)

        # pass invalid channel names
        with pytest.raises(ValueError):
            som_utils.create_pixel_matrix(fovs=fovs,
                                          channels=['chan1', 'chan2', 'chan3'],
                                          base_dir=temp_dir,
                                          tiff_dir=tiff_dir,
                                          img_sub_folder=sub_dir,
                                          seg_dir=seg_dir)

        # make the channel_norm.feather file if the test requires it
        # NOTE: pixel_mat_data already created in the previous validation tests
        if channel_norm_include:
            sample_channel_norm_df = pd.DataFrame({'channel': chans,
                                                  'norm_val': np.random.rand(len(chans))})

            feather.write_dataframe(sample_channel_norm_df,
                                    os.path.join(temp_dir, 'channel_norm.feather'),
                                    compression='uncompressed')

        # make the pixel_norm.feather file if the test requires it
        if pixel_norm_include:
            sample_pixel_norm_df = pd.DataFrame({'pixel_norm_val': np.random.rand(1)})
            feather.write_dataframe(sample_pixel_norm_df,
                                    os.path.join(temp_dir, 'pixel_norm.feather'),
                                    compression='uncompressed')

        # create the pixel matrices
        som_utils.create_pixel_matrix(fovs=fovs,
                                      channels=chans,
                                      base_dir=temp_dir,
                                      tiff_dir=tiff_dir,
                                      img_sub_folder=sub_dir,
                                      seg_dir=seg_dir)

        # check that we actually created a data directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_data'))

        # check that we actually created a subsetted directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

        # if there wasn't originally a channel_norm.json, assert one was created
        if not channel_norm_include:
            assert os.path.exists(
                os.path.join(temp_dir, 'channel_norm.feather')
            )

        # if there wasn't originally a pixel_norm.json, assert one was created
        if not pixel_norm_include:
            assert os.path.exists(
                os.path.join(temp_dir, 'pixel_norm.feather')
            )

        for fov in fovs:
            fov_data_path = os.path.join(
                temp_dir, 'pixel_mat_data', fov + '.feather'
            )
            fov_sub_path = os.path.join(
                temp_dir, 'pixel_mat_subsetted', fov + '.feather'
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
            chan_index_stop = -3 if seg_dir is None else -4
            misc_utils.verify_same_elements(
                flowsom_chans=flowsom_data_fov.columns.values[:chan_index_stop],
                provided_chans=chans
            )

            # assert no rows sum to 0
            assert np.all(flowsom_data_fov.loc[:, chans].sum(
                axis=1
            ).values != 0)

            # assert the subsetted DataFrame size is 0.1 of the preprocessed DataFrame
            # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
            assert round(flowsom_data_fov.shape[0] * 0.1) == flowsom_sub_fov.shape[0]

        # check that correct values are passed to helper function
        mocker.patch('ark.phenotyping.som_utils.create_fov_pixel_data',
                     mocked_create_fov_pixel_data)

        if sub_dir is None:
            sub_dir = ''

        # create fake data where all channels in each fov are the same
        new_tiff_dir = os.path.join(temp_dir, 'new_tiff_dir')
        os.makedirs(new_tiff_dir)
        for fov in fovs:
            img = (np.random.rand(100) * 100).reshape((10, 10))
            fov_dir = os.path.join(new_tiff_dir, fov, sub_dir)
            os.makedirs(fov_dir)
            for chan in chans:
                io.imsave(os.path.join(fov_dir, chan + '.tiff'), img)

        # create normalization file
        data_dir = os.path.join(temp_dir, 'pixel_mat_data')

        # generate the data
        mults = [1 * (1 / 2) ** i for i in range(len(chans))]

        sample_channel_norm_df = pd.DataFrame({'channel': chans,
                                               'norm_val': mults})
        feather.write_dataframe(sample_channel_norm_df,
                                os.path.join(temp_dir, 'channel_norm.feather'),
                                compression='uncompressed')

        som_utils.create_pixel_matrix(fovs=fovs,
                                      channels=chans,
                                      base_dir=temp_dir,
                                      tiff_dir=new_tiff_dir,
                                      img_sub_folder=sub_dir,
                                      seg_dir=seg_dir,
                                      dtype='float32')


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
        assert os.path.exists(os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))

        # assert the shape of norm_vals contains 1 row and number of columns = len(chan_list)
        norm_vals = feather.read_dataframe(os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))
        assert norm_vals.shape == (1, 4)

        # assert the the norm_vals columns are the same as chan_list
        misc_utils.verify_same_elements(norm_vals_channels=norm_vals.columns.values,
                                        provided_channels=chan_list)


def test_cluster_pixels(mocker):
    # basic error checks: bad path to preprocessed data folder
    # norm vals matrix, and weights matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        # bad path to preprocessed data folder
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, data_dir='bad_path'
            )

        # create a preprocessed directory for the undefined norm file test
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        # bad path to norm file
        with pytest.raises(FileNotFoundError):
            som_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, norm_vals_name='bad_path.feather'
            )

        # create a norm file for the undefined weight matrix file test
        norm_vals = pd.DataFrame(np.random.rand(1, 2), columns=['Marker1', 'Marker2'])
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))

        # bad path to weight matrix file
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
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

        for fov in fovs:
            # create dummy preprocessed data for each fov
            fov_pre_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the preprocessed data dir
            feather.write_dataframe(fov_pre_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_data',
                                                                 fov + '.feather'))

        with pytest.raises(ValueError):
            norm_vals = pd.DataFrame(
                np.random.rand(1, 4),
                columns=['Marker2', 'Marker3', 'Marker4', 'Marker5']
            )
            feather.write_dataframe(norm_vals, os.path.join(temp_dir,
                                                            'post_rowsum_chan_norm.feather'))

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
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))

        # create a dummy weights matrix and write to feather
        weights = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
        feather.write_dataframe(weights, os.path.join(temp_dir, 'pixel_weights.feather'))

        # add mocked function to "cluster" preprocessed data based on dummy weights
        mocker.patch('ark.phenotyping.som_utils.cluster_pixels', mocked_cluster_pixels)

        # run "clustering" using mocked function
        som_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_data',
                                                                   fov + '.feather'))

            # assert we didn't assign any cluster 100 or above
            cluster_ids = fov_cluster_data['pixel_som_cluster']
            assert np.all(cluster_ids < 100)


def test_pixel_consensus_cluster(mocker):
    # basic error check: bad path to data dir
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.pixel_consensus_cluster(fovs=['fov0'], channels=['chan0'],
                                              base_dir=temp_dir, data_dir='bad_path')

    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

        # create a dummy clustered matrix
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

        # store the intermediate FOV data in a dict for future comparison
        fov_data = {}

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

            # write the dummy data to pixel_mat_data
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

            fov_data[fov] = fov_cluster_matrix

        # compute averages by cluster, this happens before call to R
        cluster_avg = som_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, 'pixel_som_cluster'
        )

        # save the DataFrame
        cluster_avg.to_csv(
            os.path.join(temp_dir, 'pixel_chan_avg_som_cluster.csv'),
            index=False
        )

        # add mocked function to "consensus cluster" data averaged by cluster
        mocker.patch(
            'ark.phenotyping.som_utils.pixel_consensus_cluster',
            mocked_pixel_consensus_cluster
        )

        # run "consensus clustering" using mocked function
        som_utils.pixel_consensus_cluster(fovs=fovs, channels=chans, base_dir=temp_dir)

        for fov in fovs:
            fov_consensus_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

            # assert we didn't modify the cluster column in the consensus clustered results
            assert np.all(
                fov_data[fov]['pixel_som_cluster'].values ==
                fov_consensus_data['pixel_som_cluster'].values
            )

            # assert we didn't assign any cluster 20 or above
            consensus_cluster_ids = fov_consensus_data['pixel_meta_cluster']
            assert np.all(consensus_cluster_ids <= 20)


def test_update_pixel_meta_labels():
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

        # create a dummy data directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

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

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

        # define dummy remap schemes
        sample_pixel_remapped_dict = {i: int(i / 5) for i in np.arange(100)}
        sample_pixel_renamed_meta_dict = {i: 'meta_' + str(i) for i in sample_pixel_remapped_dict}

        # run remapping for fov0
        som_utils.update_pixel_meta_labels(
            os.path.join(temp_dir, 'pixel_mat_data'), sample_pixel_remapped_dict,
            sample_pixel_renamed_meta_dict, 'fov0'
        )

        # read remapped fov0 data in
        remapped_fov_data = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data', 'fov0.feather')
        )

        # assert the value counts of all renamed meta labels is 20
        assert np.all(remapped_fov_data['pixel_meta_cluster_rename'].value_counts().values == 50)

        # assert each meta cluster label maps to the right renamed cluster
        remapped_meta_info = dict(
            remapped_fov_data[
                ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
            ].drop_duplicates().values
        )
        for meta_cluster in remapped_meta_info:
            assert remapped_meta_info[meta_cluster] == sample_pixel_renamed_meta_dict[meta_cluster]


def test_apply_pixel_meta_cluster_remapping():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad path to pixel consensus dir
        with pytest.raises(FileNotFoundError):
            som_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'bad_consensus_dir',
                'remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy consensus dir
        os.mkdir(os.path.join(temp_dir, 'pixel_consensus_dir'))

        # basic error check: bad path to remapped name
        with pytest.raises(FileNotFoundError):
            som_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'bad_remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy remapped file
        pd.DataFrame().to_csv(os.path.join(temp_dir, 'pixel_remapping.csv'))

        # basic error check: bad path to average channel expression per SOM cluster
        with pytest.raises(FileNotFoundError):
            som_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'pixel_remapping.csv', 'bad_chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy SOM channel average file
        pd.DataFrame().to_csv(os.path.join(temp_dir, 'chan_avgs_som.csv'))

        # basic error check: bad path to average channel expression per meta cluster
        with pytest.raises(FileNotFoundError):
            som_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'pixel_remapping.csv', 'chan_avgs_som.csv', 'bad_chan_avgs_meta.csv'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        # make it easy to name metadata columns
        meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

        # create a dummy data directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

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

            # write the dummy data to pixel_mat_clustered
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data',
                                                                     fov + '.feather'))

        # define a dummy remap scheme and save
        # NOTE: we intentionally add more SOM cluster keys than necessary to show
        # that certain FOVs don't need to contain every SOM cluster available
        sample_pixel_remapping = {
            'cluster': [i for i in np.arange(105)],
            'metacluster': [int(i / 5) for i in np.arange(105)],
            'mc_name': ['meta' + str(int(i / 5)) for i in np.arange(105)]
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

        # error check: bad columns provided in the SOM to meta cluster map csv input
        with pytest.raises(ValueError):
            bad_sample_pixel_remapping = sample_pixel_remapping.copy()
            bad_sample_pixel_remapping = bad_sample_pixel_remapping.rename(
                {'mc_name': 'bad_col'},
                axis=1
            )
            bad_sample_pixel_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_pixel_remapping.csv'),
                index=False
            )

            som_utils.apply_pixel_meta_cluster_remapping(
                fovs,
                chans,
                temp_dir,
                'pixel_mat_data',
                'bad_sample_pixel_remapping.csv',
                'sample_pixel_som_cluster_chan_avgs.csv',
                'sample_pixel_meta_cluster_chan_avgs.csv'
            )

        # error check: mapping does not contain every SOM label
        with pytest.raises(ValueError):
            bad_sample_pixel_remapping = {
                'cluster': [1, 2],
                'metacluster': [1, 2],
                'mc_name': ['m1', 'm2']
            }
            bad_sample_pixel_remapping = pd.DataFrame.from_dict(bad_sample_pixel_remapping)
            bad_sample_pixel_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_pixel_remapping.csv'),
                index=False
            )

            som_utils.apply_pixel_meta_cluster_remapping(
                fovs,
                chans,
                temp_dir,
                'pixel_mat_data',
                'bad_sample_pixel_remapping.csv',
                'sample_pixel_som_cluster_chan_avgs.csv',
                'sample_pixel_meta_cluster_chan_avgs.csv'
            )

        # run the remapping process
        som_utils.apply_pixel_meta_cluster_remapping(
            fovs,
            chans,
            temp_dir,
            'pixel_mat_data',
            'sample_pixel_remapping.csv',
            'sample_pixel_som_cluster_chan_avgs.csv',
            'sample_pixel_meta_cluster_chan_avgs.csv'
        )

        # used for mapping verification
        actual_som_to_meta = sample_pixel_remapping[
            ['cluster', 'metacluster']
        ].drop_duplicates().sort_values(by='cluster')
        actual_meta_id_to_name = sample_pixel_remapping[
            ['metacluster', 'mc_name']
        ].drop_duplicates().sort_values(by='metacluster')

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
                actual_som_to_meta['cluster'].isin(som_to_meta['pixel_som_cluster'])
            ]

            assert np.all(som_to_meta.values == actual_som_to_meta_subset.values)

            # assert the mapping is the same for pixel meta cluster to renamed pixel meta cluster
            meta_id_to_name = remapped_fov_data[
                ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
            ].drop_duplicates().sort_values(by='pixel_meta_cluster')

            # this tests the case where a FOV doesn't necessarily need to have all the possible
            # meta clusters in it
            actual_meta_id_to_name_subset = actual_meta_id_to_name[
                actual_meta_id_to_name['metacluster'].isin(meta_id_to_name['pixel_meta_cluster'])
            ]

            assert np.all(meta_id_to_name.values == actual_meta_id_to_name_subset.values)

        # read in the meta cluster channel average data
        sample_pixel_channel_avg_meta_cluster = pd.read_csv(
            os.path.join(temp_dir, 'sample_pixel_meta_cluster_chan_avgs.csv')
        )

        # assert the markers data has been updated correctly
        result = np.repeat(np.array([[0.1, 0.2, 0.3, 0.4]]), repeats=20, axis=0)
        assert np.all(
            np.round(sample_pixel_channel_avg_meta_cluster[chans].values, 1) == result
        )

        # assert the counts data has been updated correctly
        assert np.all(sample_pixel_channel_avg_meta_cluster['count'].values == 150)

        # assert the correct metacluster labels are contained
        sample_pixel_channel_avg_meta_cluster = sample_pixel_channel_avg_meta_cluster.sort_values(
            by='pixel_meta_cluster'
        )
        assert np.all(sample_pixel_channel_avg_meta_cluster[
            'pixel_meta_cluster'
        ].values == np.arange(20))
        assert np.all(sample_pixel_channel_avg_meta_cluster[
            'pixel_meta_cluster_rename'
        ] == np.array(['meta' + str(i) for i in np.arange(20)]))

        # read in the som cluster channel average data
        sample_pixel_channel_avg_som_cluster = pd.read_csv(
            os.path.join(temp_dir, 'sample_pixel_som_cluster_chan_avgs.csv')
        )

        # assert the correct number of meta clusters are in and the correct number of each
        assert len(sample_pixel_channel_avg_som_cluster['pixel_meta_cluster'].value_counts()) == 20
        assert np.all(
            sample_pixel_channel_avg_som_cluster['pixel_meta_cluster'].value_counts().values == 5
        )

        # assert the correct metacluster labels are contained
        sample_pixel_channel_avg_som_cluster = sample_pixel_channel_avg_som_cluster.sort_values(
            by='pixel_meta_cluster'
        )

        assert np.all(sample_pixel_channel_avg_som_cluster[
            'pixel_meta_cluster'
        ].values == np.repeat(np.arange(20), repeats=5))
        assert np.all(sample_pixel_channel_avg_som_cluster[
            'pixel_meta_cluster_rename'
        ].values == np.array(['meta' + str(i) for i in np.repeat(np.arange(20), repeats=5)]))


def test_train_cell_som(mocker):
    # basic error check: bad path to cell table path
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.train_cell_som(
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
            som_utils.train_cell_som(
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
            som_utils.train_cell_som(
                fovs, chan_list, temp_dir, 'pixel_data_dir', cell_table_path,
                pixel_cluster_col='bad_cluster'
            )

        # TEST 1: computing weights using pixel clusters
        # compute cluster counts
        _, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_som_cluster'
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
        _, cluster_counts_norm = som_utils.create_c2pc_data(
            fovs, pixel_data_path, cell_table_path, 'pixel_meta_cluster_rename'
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
            som_utils.cluster_cells(base_dir=temp_dir, cluster_counts_norm_name='bad_path')

    # basic error check: path to cell weights does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy cluster_counts_norm_name file
        cluster_counts_norm = pd.DataFrame()
        cluster_counts_norm.to_csv(
            os.path.join(temp_dir, 'cluster_counts_norm.feather'),
            index=False
        )

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
    # basic error check: path to cell data does not exist
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            som_utils.cell_consensus_cluster(
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

            som_utils.cell_consensus_cluster(
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

            som_utils.cell_consensus_cluster(
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
            cluster_avg = som_utils.compute_cell_cluster_count_avg(
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
                'ark.phenotyping.som_utils.cell_consensus_cluster',
                mocked_cell_consensus_cluster
            )

            # "consensus cluster" the cells
            som_utils.cell_consensus_cluster(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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
            som_utils.apply_cell_meta_cluster_remapping(
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

            som_utils.apply_cell_meta_cluster_remapping(
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
        som_utils.apply_cell_meta_cluster_remapping(
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


def test_generate_meta_cluster_colormap_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad remapping path
        with pytest.raises(FileNotFoundError):
            som_utils.generate_meta_cluster_colormap_dict(
                os.path.join(temp_dir, 'bad_remap_path.csv'), None
            )

        # basic error check: remapping data contains bad columns
        with pytest.raises(ValueError):
            bad_sample_remapping = {
                'cluster': [i for i in np.arange(10)],
                'metacluster': [int(i / 50) for i in np.arange(100)],
                'mc_name_bad': ['meta' + str(int(i / 50)) for i in np.arange(100)]
            }

            bad_sample_remapping = pd.DataFrame.from_dict(bad_sample_remapping)
            bad_sample_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_remapping.csv'),
                index=False
            )

            som_utils.generate_meta_cluster_colormap_dict(
                os.path.join(temp_dir, 'bad_sample_remapping.csv'), None
            )

        # define a dummy remapping
        sample_remapping = {
            'cluster': [i for i in np.arange(100)],
            'metacluster': [int(i / 50) + 1 for i in np.arange(100)],
            'mc_name': ['meta' + str(int(i / 50) + 1) for i in np.arange(100)]
        }

        sample_remapping = pd.DataFrame.from_dict(sample_remapping)
        sample_remapping.to_csv(
            os.path.join(temp_dir, 'sample_remapping.csv'),
            index=False
        )

        # define a sample ListedColormap
        cmap = ListedColormap(['red', 'blue', 'green'])

        raw_cmap, renamed_cmap = som_utils.generate_meta_cluster_colormap_dict(
            os.path.join(temp_dir, 'sample_remapping.csv'), cmap
        )

        # assert the correct meta cluster labels are contained in both dicts
        misc_utils.verify_same_elements(
            raw_cmap_keys=list(raw_cmap.keys()),
            raw_meta_clusters=sample_remapping['metacluster'].values
        )
        misc_utils.verify_same_elements(
            renamed_cmap_keys=list(renamed_cmap.keys()),
            renamed_meta_clusters=sample_remapping['mc_name'].values
        )

        # assert the colors match up
        assert raw_cmap[1] == renamed_cmap['meta1'] == (1.0, 0.0, 0.0, 1.0)
        assert raw_cmap[2] == renamed_cmap['meta2'] == (0.0, 0.0, 1.0, 1.0)


def test_generate_weighted_channel_avg_heatmap():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad cluster channel avgs path
        with pytest.raises(FileNotFoundError):
            som_utils.generate_weighted_channel_avg_heatmap(
                os.path.join(temp_dir, 'bad_channel_avg.csv'),
                'cell_som_cluster', [], {}, {}
            )

        # basic error check: bad cell cluster col provided
        with pytest.raises(ValueError):
            dummy_chan_avg = pd.DataFrame().to_csv(
                os.path.join(temp_dir, 'sample_channel_avg.csv')
            )
            som_utils.generate_weighted_channel_avg_heatmap(
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
            som_utils.generate_weighted_channel_avg_heatmap(
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
        som_utils.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_som_cluster', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )

        # test 2: cell meta cluster channel avg
        sample_channel_avg = sample_channel_avg.drop(columns='cell_som_cluster')
        sample_channel_avg.to_csv(
            os.path.join(temp_dir, 'sample_channel_avg.csv')
        )

        # assert visualization runs
        som_utils.generate_weighted_channel_avg_heatmap(
            os.path.join(temp_dir, 'sample_channel_avg.csv'),
            'cell_meta_cluster_rename', ['chan1', 'chan2'], raw_cmap, renamed_cmap
        )


def test_add_consensus_labels_cell_table():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: cell table path does not exist
        with pytest.raises(FileNotFoundError):
            som_utils.add_consensus_labels_cell_table(
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
            som_utils.add_consensus_labels_cell_table(
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
        som_utils.add_consensus_labels_cell_table(
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
        som_utils.add_consensus_labels_cell_table(
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
