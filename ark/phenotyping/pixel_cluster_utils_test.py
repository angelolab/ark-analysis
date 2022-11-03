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
from pytest_cases import parametrize_with_cases
from skimage.draw import disk

import ark.phenotyping.pixel_cluster_utils as pixel_cluster_utils
import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

parametrize = pytest.mark.parametrize


PIXEL_MATRIX_FOVS = ['fov0', 'fov1', 'fov2']
PIXEL_MATRIX_CHANS = ['chan0', 'chan1', 'chan2']


class CreatePixelMatrixBaseCases:
    def case_all_fovs_all_chans(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, False, False

    def case_some_fovs_some_chans(self):
        return PIXEL_MATRIX_FOVS[:2], PIXEL_MATRIX_CHANS[:2], 'TIFs', True, False, False, False

    def case_no_sub_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, None, True, False, False, False

    def case_no_seg_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', False, False, False, False

    def case_existing_channel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, False, False

    def case_existing_pixel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, True, False

    def case_existing_pixel_and_channel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, True, False

    def case_new_channels_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, True, True


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


def mocked_create_fov_pixel_data(fov, channels, img_data, seg_labels, blur_factor,
                                 subset_proportion, pixel_norm_val):
    # create fake data to be compatible with downstream functions
    data = np.random.rand(len(channels) * 5).reshape(5, len(channels))
    df = pd.DataFrame(data, columns=channels)

    # hard code the metadata columns
    df['fov'] = fov
    df['row_index'] = -1
    df['column_index'] = -1
    if seg_labels is not None:
        df['seg_labels'] = -1

    # verify that each channel is 2x the previous
    for i in range(len(channels) - 1):
        assert np.allclose(img_data[..., i] * 2, img_data[..., i + 1])

    return df, df


def mocked_preprocess_fov(base_dir, tiff_dir, data_dir, subset_dir, seg_dir, seg_suffix,
                          img_sub_folder, is_mibitiff, channels, blur_factor,
                          subset_proportion, pixel_norm_val, seed, channel_norm_df, fov):
    # load img_xr from MIBITiff or directory with the fov
    if is_mibitiff:
        img_xr = load_utils.load_imgs_from_mibitiff(
            tiff_dir, mibitiff_files=[fov])
    else:
        img_xr = load_utils.load_imgs_from_tree(
            tiff_dir, img_sub_folder=img_sub_folder, fovs=[fov])

    # ensure the provided channels will actually exist in img_xr
    misc_utils.verify_in_list(
        provided_chans=channels,
        pixel_mat_chans=img_xr.channels.values
    )

    # if seg_dir is None, leave seg_labels as None
    seg_labels = None

    # otherwise, load segmentation labels in for fov
    if seg_dir is not None:
        seg_labels = io.imread(os.path.join(seg_dir, fov + seg_suffix))

    # subset for the channel data
    img_data = img_xr.loc[fov, :, :, channels].values.astype(np.float32)

    # create vector for normalizing image data
    norm_vect = channel_norm_df['norm_val'].values
    norm_vect = np.array(norm_vect).reshape([1, 1, len(norm_vect)])

    # normalize image data
    img_data = img_data / norm_vect

    # set seed for subsetting
    np.random.seed(seed)

    # create the full and subsetted fov matrices
    pixel_mat, pixel_mat_subset = mocked_create_fov_pixel_data(
        fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
        pixel_norm_val=pixel_norm_val, blur_factor=blur_factor,
        subset_proportion=subset_proportion
    )

    # write complete dataset to feather, needed for cluster assignment
    feather.write_dataframe(pixel_mat,
                            os.path.join(base_dir,
                                         data_dir,
                                         fov + ".feather"),
                            compression='uncompressed')

    # write subseted dataset to feather, needed for training
    feather.write_dataframe(pixel_mat_subset,
                            os.path.join(base_dir,
                                         subset_dir,
                                         fov + ".feather"),
                            compression='uncompressed')

    return pixel_mat


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
            assert predicted_percentiles['norm_val'].values[idx] == np.mean(percentile_dict[chan])

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
def test_filter_with_nuclear_mask(sub_dir, exclude, capsys):
    # define the fovs to use
    fovs = ['fov0', 'fov1', 'fov2']

    # define the channels to use
    chans = ['chan0', 'chan1']

    with tempfile.TemporaryDirectory() as temp_dir:
        # test seg_dir is None
        pixel_cluster_utils.filter_with_nuclear_mask(
            fovs, '', None, chans[0], ''
        )

        output = capsys.readouterr().out
        assert output == 'No seg_dir provided, you must provide one to run nuclear filtering\n'

        # test invalid seg_dir
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.filter_with_nuclear_mask(
                fovs, '', 'bad_seg_path', chans[0], ''
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
            file_name = fov + "_feature_1.tiff"
            io.imsave(os.path.join(seg_dir, file_name), rand_img,
                      check_contrast=False)

        # create sample image data
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=fovs,
            channel_names=chans, sub_dir=sub_dir, img_shape=(10, 10)
        )

        # run filtering on channel 0
        pixel_cluster_utils.filter_with_nuclear_mask(
            fovs, tiff_dir, seg_dir, 'chan0',
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
            'pixel_mat_consensus', keep_count=keep_count
        )

        # define the columns to check in cluster_avg, count may also be included
        cluster_avg_cols = chans[:]

        # verify the provided channels and the channels in cluster_avg are exactly the same
        misc_utils.verify_same_elements(
            cluster_avg_chans=cluster_avg[chans].columns.values,
            provided_chans=chans
        )

        # if keep_count is true then add the counts
        # NOTE: subtract out the corrupted counts if specified
        if keep_count:
            if cluster_col == 'pixel_som_cluster':
                counts = 20 if corrupt else 30
            else:
                counts = 200 if corrupt else 300

            count_col = np.expand_dims(np.repeat(counts, repeats=result.shape[0]), axis=1)
            result = np.append(result, count_col, 1)

            cluster_avg_cols.append('count')

        # assert all elements of cluster_avg and the actual result are equal
        assert np.array_equal(result, np.round(cluster_avg[cluster_avg_cols].values, 1))


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
        sample_pixel_mat, sample_pixel_mat_subset = pixel_cluster_utils.create_fov_pixel_data(
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
        sample_pixel_mat, sample_pixel_mat_subset = pixel_cluster_utils.create_fov_pixel_data(
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
            file_name = fov + "_feature_0.tiff"
            io.imsave(os.path.join(seg_dir, file_name), rand_img,
                      check_contrast=False)

        # generate sample channel normalization values
        channel_norm_df = pd.DataFrame.from_dict({
            'channel': chans,
            'norm_val': np.repeat(10, repeats=len(chans))
        })

        # run the preprocessing for fov0
        # NOTE: don't test the return value, leave that for test_create_pixel_matrix
        pixel_cluster_utils.preprocess_fov(
            temp_dir, tiff_dir, 'pixel_mat_data', 'pixel_mat_subsetted',
            seg_dir, '_feature_0.tiff', 'TIFs', False, ['chan0', 'chan1', 'chan2'],
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


# TODO: leaving out MIBItiff testing until someone needs it
@parametrize_with_cases(
    'fovs,chans,sub_dir,seg_dir_include,channel_norm_include,pixel_norm_include,norm_diff_chan',
    cases=CreatePixelMatrixBaseCases
)
@parametrize('multiprocess', [True, False])
def test_create_pixel_matrix_base(fovs, chans, sub_dir, seg_dir_include,
                                  channel_norm_include, pixel_norm_include,
                                  norm_diff_chan, multiprocess, mocker, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a directory to store the image data
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)

        # invalid subset proportion specified
        with pytest.raises(ValueError):
            pixel_cluster_utils.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                seg_dir=None,
                subset_proportion=1.1
            )

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir='bad_base_dir',
                tiff_dir=tiff_dir,
                seg_dir=None
            )

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir='bad_tiff_dir',
                seg_dir=None
            )

        # pass invalid pixel output directory
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir=temp_dir,
                seg_dir=None,
                pixel_output_dir='bad_output_dir'
            )

        # make a dummy pixel_output_dir
        sample_pixel_output_dir = os.path.join(temp_dir, 'pixel_output_dir')
        os.mkdir(sample_pixel_output_dir)

        # create a dummy seg_dir with data if we're on a test that requires segmentation labels
        if seg_dir_include:
            seg_dir = os.path.join(temp_dir, 'segmentation')
            os.mkdir(seg_dir)

            # create sample segmentation data
            for fov in fovs:
                rand_img = np.random.randint(0, 16, size=(10, 10))
                file_name = fov + "_feature_0.tiff"
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
            pixel_cluster_utils.create_pixel_matrix(
                fovs=['fov1', 'fov2', 'fov3'],
                channels=chans,
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # pass invalid channel names
        with pytest.raises(ValueError):
            pixel_cluster_utils.create_pixel_matrix(
                fovs=fovs,
                channels=['chan1', 'chan2', 'chan3'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # make the channel_norm.feather file if the test requires it
        # NOTE: pixel_mat_data already created in the previous validation tests
        if channel_norm_include:
            # helps test if channel_norm.feather contains a different set of channels
            norm_chans = [chans[0]] if norm_diff_chan else chans
            sample_channel_norm_df = pd.DataFrame({'channel': norm_chans,
                                                  'norm_val': np.random.rand(len(norm_chans))})

            feather.write_dataframe(
                sample_channel_norm_df,
                os.path.join(temp_dir, sample_pixel_output_dir, 'test_channel_norm.feather'),
                compression='uncompressed'
            )

        # make the pixel_norm.feather file if the test requires it
        if pixel_norm_include:
            sample_pixel_norm_df = pd.DataFrame({'pixel_norm_val': np.random.rand(1)})
            feather.write_dataframe(
                sample_pixel_norm_df,
                os.path.join(temp_dir, sample_pixel_output_dir, 'test_pixel_norm.feather'),
                compression='uncompressed'
            )

        # create the pixel matrices
        pixel_cluster_utils.create_pixel_matrix(
            fovs=fovs,
            channels=chans,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=sub_dir,
            seg_dir=seg_dir,
            pixel_cluster_prefix='test',
            multiprocess=multiprocess
        )

        # assert we overwrote the original channel_norm and pixel_norm files
        # if new set of channels provided
        if norm_diff_chan:
            output_capture = capsys.readouterr().out
            assert 'New channels provided: overwriting whole cohort' in output_capture

        # check that we actually created a data directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_data'))

        # check that we actually created a subsetted directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

        # if there wasn't originally a channel_norm.feather or if overwritten, assert one created
        if not channel_norm_include or norm_diff_chan:
            assert os.path.exists(
                os.path.join(temp_dir, sample_pixel_output_dir, 'test_channel_norm.feather')
            )

        # if there wasn't originally a pixel_norm.feather or if overwritten, assert one created
        if not pixel_norm_include or norm_diff_chan:
            assert os.path.exists(
                os.path.join(temp_dir, sample_pixel_output_dir, 'test_pixel_norm.feather')
            )

        for fov in fovs:
            fov_data_path = os.path.join(
                temp_dir, 'pixel_mat_data', fov + '.feather'
            )
            fov_sub_path = os.path.join(
                temp_dir, 'pixel_mat_subsetted', fov + '.feather'
            )

            # assert we actually created a .feather preprocessed file for each fov
            assert os.path.exists(fov_data_path)

            # assert that we actually created a .feather subsetted file for each fov
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
        mocker.patch(
            'ark.phenotyping.pixel_cluster_utils.preprocess_fov',
            mocked_preprocess_fov
        )

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

        # recreate the output directory
        rmtree(sample_pixel_output_dir)
        os.mkdir(sample_pixel_output_dir)

        # create normalization file
        data_dir = os.path.join(temp_dir, 'pixel_mat_data')

        # generate the data
        mults = [(1 / 2) ** i for i in range(len(chans))]

        sample_channel_norm_df = pd.DataFrame({'channel': chans,
                                               'norm_val': mults})
        feather.write_dataframe(
            sample_channel_norm_df,
            os.path.join(temp_dir, sample_pixel_output_dir, 'test_channel_norm.feather'),
            compression='uncompressed'
        )

        pixel_cluster_utils.create_pixel_matrix(
            fovs=fovs,
            channels=chans,
            base_dir=temp_dir,
            tiff_dir=new_tiff_dir,
            img_sub_folder=sub_dir,
            seg_dir=seg_dir,
            pixel_cluster_prefix='test',
            multiprocess=multiprocess
        )


# TODO: clean up the following tests
def generate_create_pixel_matrix_test_data(temp_dir):
    # create a directory to store the image data
    tiff_dir = os.path.join(temp_dir, 'sample_image_data')
    os.mkdir(tiff_dir)

    # create sample image data
    test_utils.create_paired_xarray_fovs(
        base_dir=tiff_dir, fov_names=PIXEL_MATRIX_FOVS,
        channel_names=PIXEL_MATRIX_CHANS, sub_dir=None, img_shape=(10, 10)
    )

    # make a sample pixel_output_dir
    os.mkdir(os.path.join(temp_dir, 'pixel_output_dir'))

    # create the data, this is just for generation
    pixel_cluster_utils.create_pixel_matrix(
        fovs=PIXEL_MATRIX_FOVS,
        channels=PIXEL_MATRIX_CHANS,
        base_dir=temp_dir,
        tiff_dir=tiff_dir,
        img_sub_folder=None,
        seg_dir=None
    )


@parametrize('multiprocess', [True, False])
def test_create_pixel_matrix_missing_fov(multiprocess, capsys):
    fov_files = [fov + '.feather' for fov in PIXEL_MATRIX_FOVS]

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_create_pixel_matrix_test_data(temp_dir)
        capsys.readouterr()

        tiff_dir = os.path.join(temp_dir, 'sample_image_data')

        # test the case where we've already written FOVs to both data and subset folder
        os.remove(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'))
        os.remove(os.path.join(temp_dir, 'pixel_mat_subsetted', 'fov1.feather'))
        sample_quant_data = pd.DataFrame(
            np.random.rand(3, 2),
            index=PIXEL_MATRIX_CHANS,
            columns=['fov0', 'fov2']
        )
        feather.write_dataframe(
            sample_quant_data,
            os.path.join(temp_dir, 'pixel_output_dir', 'quant_dat.feather')
        )

        pixel_cluster_utils.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None,
            multiprocess=multiprocess
        )

        output_capture = capsys.readouterr().out
        assert output_capture == (
            "Restarting preprocessing from FOV fov1, 1 fovs left to process\n"
            "Processed 1 fovs\n"
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )

        capsys.readouterr()

        # test the case where we've written a FOV to data but not subset
        # NOTE: in this case, the value in quant_dat will also not have been written
        os.remove(os.path.join(temp_dir, 'pixel_mat_subsetted', 'fov1.feather'))
        feather.write_dataframe(
            sample_quant_data,
            os.path.join(temp_dir, 'pixel_output_dir', 'quant_dat.feather')
        )

        pixel_cluster_utils.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None,
            multiprocess=multiprocess
        )

        output_capture = capsys.readouterr().out
        assert output_capture == (
            "Restarting preprocessing from FOV fov1, 1 fovs left to process\n"
            "Processed 1 fovs\n"
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )


def test_create_pixel_matrix_all_fovs(capsys):
    fov_files = [fov + '.feather' for fov in PIXEL_MATRIX_FOVS]

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_create_pixel_matrix_test_data(temp_dir)
        capsys.readouterr()

        tiff_dir = os.path.join(temp_dir, 'sample_image_data')

        pixel_cluster_utils.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None
        )

        output_capture = capsys.readouterr().out
        assert output_capture == "There are no more FOVs to preprocess, skipping\n"
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )


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


def test_train_pixel_som(mocker):
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.train_pixel_som(
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

        # not all of the provided fovs exist
        with pytest.raises(ValueError):
            pixel_cluster_utils.train_pixel_som(
                fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir
            )

        # column mismatch between provided channels and subsetted data
        with pytest.raises(ValueError):
            pixel_cluster_utils.train_pixel_som(
                fovs=fovs, channels=['Marker3', 'Marker4', 'MarkerBad'],
                base_dir=temp_dir
            )

        # add mocked function to "train" the SOM based on dummy subsetted data
        mocker.patch('ark.phenotyping.pixel_cluster_utils.train_pixel_som', mocked_train_pixel_som)

        # run "training" using mocked function
        pixel_cluster_utils.train_pixel_som(fovs=fovs, channels=chan_list, base_dir=temp_dir)

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
            pixel_cluster_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, data_dir='bad_path'
            )

        # create a preprocessed directory for the undefined norm file test
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_preprocessed'))

        # bad path to norm file
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.cluster_pixels(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, norm_vals_name='bad_path.feather'
            )

        # create a norm file for the undefined weight matrix file test
        norm_vals = pd.DataFrame(np.random.rand(1, 2), columns=['Marker1', 'Marker2'])
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))

        # bad path to weight matrix file
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.cluster_pixels(
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
            pixel_cluster_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

            # column name ordering mismatch for weights
            pixel_cluster_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

            # not all the provided fovs exist
            pixel_cluster_utils.cluster_pixels(
                fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir
            )

        # create a dummy normalized values matrix and write to feather
        norm_vals = pd.DataFrame(np.ones((1, 4)), columns=chan_list)
        feather.write_dataframe(norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather'))

        # create a dummy weights matrix and write to feather
        weights = pd.DataFrame(np.random.rand(100, 4), columns=chan_list)
        feather.write_dataframe(weights, os.path.join(temp_dir, 'pixel_weights.feather'))

        # add mocked function to "cluster" preprocessed data based on dummy weights
        mocker.patch('ark.phenotyping.pixel_cluster_utils.cluster_pixels', mocked_cluster_pixels)

        # run "clustering" using mocked function
        pixel_cluster_utils.cluster_pixels(fovs=fovs, channels=chan_list, base_dir=temp_dir)

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
            pixel_cluster_utils.pixel_consensus_cluster(
                fovs=['fov0'], channels=['chan0'],
                base_dir=temp_dir, data_dir='bad_path'
            )

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
        cluster_avg = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
            fovs, chans, temp_dir, 'pixel_som_cluster'
        )

        # save the DataFrame
        cluster_avg.to_csv(
            os.path.join(temp_dir, 'pixel_chan_avg_som_cluster.csv'),
            index=False
        )

        # add mocked function to "consensus cluster" data averaged by cluster
        mocker.patch(
            'ark.phenotyping.pixel_cluster_utils.pixel_consensus_cluster',
            mocked_pixel_consensus_cluster
        )

        # run "consensus clustering" using mocked function
        pixel_cluster_utils.pixel_consensus_cluster(fovs=fovs, channels=chans, base_dir=temp_dir)

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

        # if specified, write just fov0 to pixel_mat_data_temp
        if generate_temp and fov == 'fov0':
            # append a dummy rename column
            fov_cluster_matrix['pixel_meta_cluster_rename'] = np.repeat(np.arange(10), repeats=100)
            feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                     'pixel_mat_data_temp',
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


# TODO: split up this test function
@parametrize('multiprocess', [True, False])
def test_apply_pixel_meta_cluster_remapping_base(multiprocess):
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad path to pixel consensus dir
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'bad_consensus_dir',
                'remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy consensus dir
        os.mkdir(os.path.join(temp_dir, 'pixel_consensus_dir'))

        # basic error check: bad path to remapped name
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'bad_remapped_name.csv', 'chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy remapped file
        pd.DataFrame().to_csv(os.path.join(temp_dir, 'pixel_remapping.csv'))

        # basic error check: bad path to average channel expression per SOM cluster
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'pixel_remapping.csv', 'bad_chan_avgs_som.csv', 'chan_avgs_meta.csv'
            )

        # make a dummy SOM channel average file
        pd.DataFrame().to_csv(os.path.join(temp_dir, 'chan_avgs_som.csv'))

        # basic error check: bad path to average channel expression per meta cluster
        with pytest.raises(FileNotFoundError):
            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
                ['fov0'], ['chan0'], temp_dir, 'pixel_consensus_dir',
                'pixel_remapping.csv', 'chan_avgs_som.csv', 'bad_chan_avgs_meta.csv'
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
                {'mc_name': 'bad_col'},
                axis=1
            )
            bad_sample_pixel_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_pixel_remapping.csv'),
                index=False
            )

            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
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

            pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
                fovs,
                chans,
                temp_dir,
                'pixel_mat_data',
                'bad_sample_pixel_remapping.csv',
                'sample_pixel_som_cluster_chan_avgs.csv',
                'sample_pixel_meta_cluster_chan_avgs.csv'
            )

        # run the remapping process
        pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
            fovs,
            chans,
            temp_dir,
            'pixel_mat_data',
            'sample_pixel_remapping.csv',
            'sample_pixel_som_cluster_chan_avgs.csv',
            'sample_pixel_meta_cluster_chan_avgs.csv',
            multiprocess=multiprocess
        )

        # assert _temp dir no longer exists (pixel_mat_data_temp should be renamed pixel_mat_data)
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

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
        pixel_cluster_utils.apply_pixel_meta_cluster_remapping(
            fovs,
            chans,
            temp_dir,
            'pixel_mat_data',
            'sample_pixel_remapping.csv',
            'sample_pixel_som_cluster_chan_avgs.csv',
            'sample_pixel_meta_cluster_chan_avgs.csv',
            multiprocess=multiprocess
        )

        # assert the _temp folder is now gone
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        output = capsys.readouterr().out
        print(output)
        desired_status_updates = "Using re-mapping scheme to re-label pixel meta clusters\n"
        desired_status_updates += "The data for FOV fov1 has been corrupted, skipping\n"
        assert desired_status_updates in output

        # verify that the FOVs in pixel_mat_data are correct
        # NOTE: fov1 should not be written because it was corrupted
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=['fov0.feather', 'fov2.feather']
        )
