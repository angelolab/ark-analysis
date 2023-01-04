from functools import partial
import multiprocessing
import os
from shutil import rmtree
import subprocess
import warnings

import feather
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
import random
import scipy.ndimage as ndimage
from skimage.io import imread, imsave

from ark.phenotyping import cluster_helpers
from ark.utils import io_utils, load_utils, misc_utils

multiprocessing.set_start_method('spawn', force=True)


def calculate_channel_percentiles(tiff_dir, fovs, channels, img_sub_folder,
                                  percentile):
    """Calculates average percentile for each channel in the dataset

    Args:
        tiff_dir (str):
            Name of the directory containing the tiff files
        fovs (list):
            List of fovs to include
        channels (list):
            List of channels to include
        img_sub_folder (str):
            Sub folder within each FOV containing image data
        percentile (float):
            The specific percentile to compute

    Returns:
        pd.DataFrame:
            The mapping between each channel and its normalization value
    """

    # create list to hold percentiles
    percentile_means = []

    # loop over channels and FOVs
    for channel in channels:
        percentile_list = []
        for fov in fovs:
            # load image data and remove 0 valued pixels
            img = load_utils.load_imgs_from_tree(data_dir=tiff_dir, img_sub_folder=img_sub_folder,
                                                 channels=[channel], fovs=[fov]).values[0, :, :, 0]
            img = img[img > 0]

            # record and store percentile, skip if no non-zero pixels
            if len(img) > 0:
                img_percentile = np.quantile(img, percentile)
                percentile_list.append(img_percentile)

        # save channel-wide average
        percentile_means.append(np.mean(percentile_list))

    percentile_df = pd.DataFrame(np.expand_dims(percentile_means, axis=0), columns=channels)

    return percentile_df


def calculate_pixel_intensity_percentile(tiff_dir, fovs, channels, img_sub_folder,
                                         channel_percentiles, percentile=0.05):
    """Calculates average percentile per FOV for total signal in each pixel

    Args:
        tiff_dir (str):
            Name of the directory containing the tiff files
        fovs (list):
            List of fovs to include
        channels (list):
            List of channels to include
        img_sub_folder (str):
            Sub folder within each FOV containing image data
        channel_percentiles (pd.DataFrame):
            The mapping between each channel and its normalization value
            Computed by `calculate_channel_percentiles`
        percentile (float):
            The pixel intensity percentile per FOV to average over


    Returns:
        float:
            The average percentile per FOV for total signal in each pixel
    """

    # create vector of channel percentiles to enable broadcasting
    norm_vect = channel_percentiles.iloc[0].values
    norm_vect = norm_vect.reshape([1, 1, len(norm_vect)])

    intensity_percentile_list = []

    for fov in fovs:
        # load image data
        img_data = load_utils.load_imgs_from_tree(data_dir=tiff_dir, fovs=[fov],
                                                  channels=channels, img_sub_folder=img_sub_folder)

        # normalize each channel by its percentile value
        norm_data = img_data[0].values / norm_vect

        # sum channels together to determine total intensity
        summed_data = np.sum(norm_data, axis=-1)
        intensity_percentile_list.append(np.quantile(summed_data, percentile))

    return np.mean(intensity_percentile_list)


def normalize_rows(pixel_data, channels, include_seg_label=True):
    """Normalizes the rows of a pixel matrix by their sum

    Args:
        pixel_data (pandas.DataFrame):
            The dataframe containing the pixel data for a given fov
            Includes channel and meta (`fov`, `segmentation_label`, etc.) columns
        channels (list):
            List of channels to subset over
        include_seg_label (bool):
            Whether to include `'segmentation_label'` as a metadata column

    Returns:
        pandas.DataFrame:
            The pixel data with rows normalized and 0-sum rows removed
    """

    # subset the fov data by the channels the user trained the pixel SOM on
    pixel_data_sub = pixel_data[channels]

    # divide each row by their sum
    pixel_data_sub = pixel_data_sub.div(pixel_data_sub.sum(axis=1), axis=0)

    # define the meta columns to add back
    meta_cols = ['fov', 'row_index', 'column_index']

    # add the segmentation_label column if it should be kept
    if include_seg_label:
        meta_cols.append('segmentation_label')

    # add back meta columns, making sure to remove 0-row indices
    pixel_data_sub[meta_cols] = pixel_data.loc[pixel_data_sub.index.values, meta_cols]

    return pixel_data_sub


def check_for_modified_channels(tiff_dir, test_fov, img_sub_folder, channels):
    """Checks to make sure the user selected newly modified channels

    Args:
        tiff_dir (str):
            Name of the directory containing the tiff files
        test_fov (str):
            example fov used to check channel names
        img_sub_folder (str):
            sub-folder within each FOV containing image data
        channels (list):
            list of channels to use for analysis
    """

    # convert to path-compatible format
    if img_sub_folder is None:
        img_sub_folder = ''

    # get all channels within example FOV
    all_channels = io_utils.list_files(os.path.join(tiff_dir, test_fov, img_sub_folder))
    all_channels = io_utils.remove_file_extensions(all_channels)
    # define potential modifications to channel names
    mods = ['_smoothed', '_nuc_include', '_nuc_exclude']

    # loop over each user-provided channel
    for channel in channels:
        for mod in mods:
            # check for substring matching
            chan_mod = channel + mod
            if chan_mod in all_channels:
                warnings.warn('You selected {} as the channel to analyze, but there were potential'
                              ' modified channels found: {}. Make sure you selected the correct '
                              'version of the channel for inclusion in '
                              'clustering'.format(channel, chan_mod))
            else:
                pass


def smooth_channels(fovs, tiff_dir, img_sub_folder, channels, smooth_vals):
    """Adds additional smoothing for selected channels as a preprocessing step

    Args:
        fovs (list):
            List of fovs to process
        tiff_dir (str):
            Name of the directory containing the tiff files
        img_sub_folder (str):
            sub-folder within each FOV containing image data
        channels (list):
            list of channels to apply smoothing to
        smooth_vals (list or int):
            amount to smooth channels. If a single int, applies
            to all channels. Otherwise, a custom value per channel can be supplied

    """

    # no output if no channels specified
    if channels is None or len(channels) == 0:
        return

    # convert to path-compatible format
    if img_sub_folder is None:
        img_sub_folder = ''

    # convert int to list of same length
    if type(smooth_vals) is int:
        smooth_vals = [smooth_vals for _ in range(len(channels))]
    elif type(smooth_vals) is list:
        if len(smooth_vals) != len(channels):
            raise ValueError("A list was provided for variable smooth_vals, but it does not "
                             "have the same length as the list of channels provided")
    else:
        raise ValueError("Variable smooth_vals must be either a single integer or a list")

    for fov in fovs:
        for idx, chan in enumerate(channels):
            img = load_utils.load_imgs_from_tree(data_dir=tiff_dir, img_sub_folder=img_sub_folder,
                                                 fovs=[fov], channels=[chan]).values[0, :, :, 0]
            chan_out = ndimage.gaussian_filter(img, sigma=smooth_vals[idx])
            imsave(os.path.join(tiff_dir, fov, img_sub_folder, chan + '_smoothed.tiff'),
                   chan_out, check_contrast=False)


def filter_with_nuclear_mask(fovs, tiff_dir, seg_dir, channel,
                             img_sub_folder=None, exclude=True):
    """Filters out background staining using subcellular marker localization.

    Non-nuclear signal is removed from nuclear markers and vice-versa for membrane markers.

    Args:
        fovs (list):
            The list of fovs to filter
        tiff_dir (str):
            Name of the directory containing the tiff files
        seg_dir (str):
            Name of the directory containing the segmented files
        channel (str):
            Channel to apply filtering to
        img_sub_folder (str):
            Name of the subdirectory inside `tiff_dir` containing the tiff files.
            Set to `None` if there isn't any.
        exclude (bool):
            Whether to filter out nuclear or membrane signal
    """

    # if seg_dir is None, the user cannot run filtering
    if seg_dir is None:
        print('No seg_dir provided, you must provide one to run nuclear filtering')
        return

    # raise an error if the provided seg_dir does not exist
    io_utils.validate_paths(seg_dir)

    # convert to path-compatible format
    if img_sub_folder is None:
        img_sub_folder = ''

    for fov in fovs:
        # load the channel image in
        img = load_utils.load_imgs_from_tree(data_dir=tiff_dir, img_sub_folder=img_sub_folder,
                                             fovs=[fov], channels=[channel]).values[0, :, :, 0]

        # load the segmented image in
        seg_img = imread(os.path.join(seg_dir, fov + '_nuclear.tiff'))[0, ...]

        # mask out the nucleus
        if exclude:
            suffix = '_nuc_exclude.tiff'
            seg_mask = seg_img > 0
        # mask out the membrane
        else:
            suffix = '_nuc_include.tiff'
            seg_mask = seg_img == 0

        # filter out the nucleus or membrane depending on exclude parameter
        img[seg_mask] = 0

        # save filtered image
        imsave(os.path.join(tiff_dir, fov, img_sub_folder, channel + suffix), img,
               check_contrast=False)


def compute_pixel_cluster_channel_avg(fovs, channels, base_dir, pixel_cluster_col,
                                      num_pixel_clusters,
                                      pixel_data_dir='pixel_mat_data',
                                      num_fovs_subset=100, seed=42, keep_count=False):
    """Compute the average channel values across each pixel SOM cluster.

    To improve performance, number of FOVs is downsampled by `fov_subset_proportion`

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_cluster_col (str):
            Name of the column to group by
        num_pixel_clusters (int):
            The number of pixel clusters that are desired
        pixel_data_dir (str):
            Name of the directory containing the pixel data with cluster labels
        num_fovs_subset (float):
            The number of FOVs to subset on. Note that if `len(fovs) < num_fovs_subset`, all of
            the FOVs will still be selected
        seed (int):
            The random seed to use for subsetting FOVs
        keep_count (bool):
            Whether to keep the count column when aggregating or not
            This should only be set to `True` for visualization purposes

    Returns:
        pandas.DataFrame:
            Contains the average channel values for each pixel SOM/meta cluster
    """

    # verify the pixel cluster col specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster']
    )

    # verify num_pixel_clusters is valid
    if num_pixel_clusters <= 0:
        raise ValueError("Number of pixel clusters desired must be a positive integer")

    # verify fovs subset value is valid
    if num_fovs_subset <= 0:
        raise ValueError("Number of fovs to subset must be a positive integer")

    # define a list to hold the cluster averages for each FOV
    fov_cluster_avgs = []

    # warn the user that we can only select so many FOVs if len(fovs) < num_fovs_subset
    if len(fovs) < num_fovs_subset:
        warnings.warn(
            'Provided num_fovs_subset=%d but only %d FOVs in dataset, '
            'subsetting just the %d FOVs' %
            (num_fovs_subset, len(fovs), len(fovs))
        )

    # subset number of FOVs based on num_fovs_subset if less than total number of fovs
    random.seed(seed)
    fovs_sub = random.sample(fovs, num_fovs_subset) if num_fovs_subset < len(fovs) else fovs

    for fov in fovs_sub:
        # read in the fovs data
        try:
            fov_pixel_data = feather.read_dataframe(
                os.path.join(base_dir, pixel_data_dir, fov + '.feather')
            )
        except (ArrowInvalid, OSError, IOError):
            print("The data for FOV %s has been corrupted, skipping" % fov)
            continue

        # aggregate the sums and counts
        sum_by_cluster = fov_pixel_data.groupby(
            pixel_cluster_col
        )[channels].sum()
        count_by_cluster = fov_pixel_data.groupby(
            pixel_cluster_col
        )[channels].size().to_frame('count')

        # merge the results by column
        agg_results = pd.merge(
            sum_by_cluster, count_by_cluster, left_index=True, right_index=True
        ).reset_index()

        # append the result to cluster_avgs
        fov_cluster_avgs.append(agg_results)

    cluster_avgs = pd.concat(fov_cluster_avgs)

    # reset the index of cluster_avgs for consistency
    cluster_avgs = cluster_avgs.reset_index(drop=True)

    # sum the counts and the channel sums
    sum_count_totals = cluster_avgs.groupby(
        pixel_cluster_col
    )[channels + ['count']].sum().reset_index()

    # error out if any clusters were lost during the averaging process
    if sum_count_totals.shape[0] < num_pixel_clusters:
        raise ValueError(
            'Averaged data contains just %d clusters out of %d. '
            'Average expression file not written. '
            'Consider increasing your num_fovs_subset value.' %
            (sum_count_totals.shape[0], num_pixel_clusters)
        )

    # now compute the means using the count column
    sum_count_totals[channels] = sum_count_totals[channels].div(sum_count_totals['count'], axis=0)

    # convert cluster column to integer type
    sum_count_totals[pixel_cluster_col] = sum_count_totals[pixel_cluster_col].astype(int)

    # sort cluster col in ascending order
    sum_count_totals = sum_count_totals.sort_values(by=pixel_cluster_col)

    # drop the count column if specified
    if not keep_count:
        sum_count_totals = sum_count_totals.drop('count', axis=1)

    return sum_count_totals


def create_fov_pixel_data(fov, channels, img_data, seg_labels, pixel_thresh_val,
                          blur_factor=2, subset_proportion=0.1):
    """Preprocess pixel data for one fov

    Args:
        fov (str):
            Name of the fov to index
        channels (list):
            List of channels to subset over
        img_data (numpy.ndarray):
            Array representing image data for one fov
        seg_labels (numpy.ndarray):
            Array representing segmentation labels for one fov
        pixel_thresh_val (float):
            value used to determine per-pixel cutoff for total signal inclusion
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov

    Returns:
        tuple:
            Contains the following:

            - `pandas.DataFrame`: Gaussian blurred and channel sum normalized pixel data for a fov
            - `pandas.DataFrame`: subset of the preprocessed pixel dataset for a fov
    """

    # for each marker, compute the Gaussian blur
    for marker in range(len(channels)):
        img_data[:, :, marker] = ndimage.gaussian_filter(img_data[:, :, marker],
                                                         sigma=blur_factor)

    # flatten each image, make sure to subset only on channels
    pixel_mat = img_data.reshape(-1, len(channels))

    # convert into a dataframe
    pixel_mat = pd.DataFrame(pixel_mat, columns=channels)

    # assign metadata about each entry
    pixel_mat['fov'] = fov
    pixel_mat['row_index'] = np.repeat(range(img_data.shape[0]), img_data.shape[1])
    pixel_mat['column_index'] = np.tile(range(img_data.shape[1]), img_data.shape[0])

    # assign segmentation labels if it is not None
    if seg_labels is not None:
        seg_labels_flat = seg_labels.flatten()
        pixel_mat['segmentation_label'] = seg_labels_flat

    # remove any rows with channels with a sum below the threshold
    rowsums = pixel_mat[channels].sum(axis=1)
    pixel_mat = pixel_mat.loc[rowsums > pixel_thresh_val, :].reset_index(drop=True)

    # normalize the row sums of pixel mat
    pixel_mat = normalize_rows(pixel_mat, channels, seg_labels is not None)

    # subset the pixel matrix for training
    pixel_mat_subset = pixel_mat.sample(frac=subset_proportion)

    return pixel_mat, pixel_mat_subset


def preprocess_fov(base_dir, tiff_dir, data_dir, subset_dir, seg_dir, seg_suffix,
                   img_sub_folder, is_mibitiff, channels, blur_factor,
                   subset_proportion, pixel_thresh_val, seed, channel_norm_df, fov):
    """Helper function to read in the FOV-level pixel data, run `create_fov_pixel_data`,
    and save the preprocessed data.

    Args:
        base_dir (str):
            The path to the data directories
        tiff_dir (str):
            Name of the directory containing the tiff files
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data
        subset_dir (str):
            The name of the directory containing the subsetted pixel data
        seg_dir (str):
            Name of the directory containing the segmented files.
            Set to `None` if no segmentation directory is available or desired.
        seg_suffix (str):
            The suffix that the segmentation images use.
            Ignored if `seg_dir` is `None`.
        img_sub_folder (str):
            Name of the subdirectory inside `tiff_dir` containing the tiff files.
            Set to `None` if there isn't any.
        is_mibitiff (bool):
            Whether to load the images from MIBITiff
        channels (list):
            List of channels to subset over, applies only to `pixel_mat_subset`
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov
        pixel_thresh_val (float):
            The value to normalize the pixels by
        seed (int):
            The random seed to set for subsetting
        channel_norm_df (pandas.DataFrame):
            The channel normalization values to use
        fov (str):
            The name of the FOV to preprocess

    Returns:
        pandas.DataFrame:
            The full preprocessed pixel dataset, needed for computing
            99.9% normalized values in `create_pixel_matrix`
    """

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
        seg_labels = imread(os.path.join(seg_dir, fov + seg_suffix))

    # subset for the channel data
    img_data = img_xr.loc[fov, :, :, channels].values.astype(np.float32)

    # create vector for normalizing image data
    norm_vect = channel_norm_df.iloc[0].values
    norm_vect = np.array(norm_vect).reshape([1, 1, len(norm_vect)])

    # normalize image data
    img_data = img_data / norm_vect

    # set seed for subsetting
    np.random.seed(seed)

    # create the full and subsetted fov matrices
    pixel_mat, pixel_mat_subset = create_fov_pixel_data(
        fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
        pixel_thresh_val=pixel_thresh_val, blur_factor=blur_factor,
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


def create_pixel_matrix(fovs, channels, base_dir, tiff_dir, seg_dir,
                        img_sub_folder="TIFs", seg_suffix='_whole_cell.tiff',
                        pixel_output_dir='pixel_output_dir',
                        data_dir='pixel_mat_data',
                        subset_dir='pixel_mat_subsetted',
                        norm_vals_name='channel_norm_post_rowsum.feather', is_mibitiff=False,
                        blur_factor=2, subset_proportion=0.1, seed=42,
                        channel_percentile=0.99, multiprocess=False, batch_size=5):
    """For each fov, add a Gaussian blur to each channel and normalize channel sums for each pixel

    Saves data to `data_dir` and subsetted data to `subset_dir`

    Args:
        fovs (list):
            List of fovs to subset over
        channels (list):
            List of channels to subset over, applies only to `pixel_mat_subset`
        base_dir (str):
            The path to the data directories
        tiff_dir (str):
            Name of the directory containing the tiff files
        seg_dir (str):
            Name of the directory containing the segmented files.
            Set to `None` if no segmentation directory is available or desired.
        img_sub_folder (str):
            Name of the subdirectory inside `tiff_dir` containing the tiff files.
            Set to `None` if there isn't any.
        seg_suffix (str):
            The suffix that the segmentation images use.
            Ignored if `seg_dir` is `None`.
        pixel_output_dir (str):
            The name of the data directory containing the pixel data to use for the
            clustering pipeline. `data_dir` and `subset_dir` should be placed here.
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data.
            Should be placed in `pixel_output_dir`.
        subset_dir (str):
            The name of the directory containing the subsetted pixel data.
            Should be placed in `pixel_output_dir`.
        norm_vals_name (str):
            The name of the file to store the 99.9% normalization values
        is_mibitiff (bool):
            Whether to load the images from MIBITiff
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov
        seed (int):
            The random seed to set for subsetting
        channel_percentile (float):
            Percentile used to normalize channels to same range
        multiprocess (bool):
            Whether to use multiprocessing or not
        batch_size (int):
            The number of FOVs to process in parallel, ignored if `multiprocess` is `False`
    """

    # if the subset_proportion specified is out of range
    if subset_proportion <= 0 or subset_proportion > 1:
        raise ValueError('Invalid subset percentage entered: must be in (0, 1]')

    # path validation
    io_utils.validate_paths([base_dir, tiff_dir, os.path.join(base_dir, pixel_output_dir)])

    # create data_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, data_dir)):
        os.mkdir(os.path.join(base_dir, data_dir))

    # create subset_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, subset_dir)):
        os.mkdir(os.path.join(base_dir, subset_dir))

    # define path to channel normalization values
    channel_norm_path = os.path.join(
        base_dir, pixel_output_dir, 'channel_norm.feather'
    )

    # define path to pixel normalization values
    pixel_thresh_path = os.path.join(
        base_dir, pixel_output_dir, 'pixel_thresh.feather'
    )

    # reset entire cohort if channels provided are different from ones in existing channel_norm
    if os.path.exists(channel_norm_path):
        channel_norm_df = feather.read_dataframe(channel_norm_path)

        if set(channel_norm_df.columns.values) != set(channels):
            print("New channels provided: overwriting whole cohort")

            # delete the existing data in data_dir and subset_dir
            rmtree(os.path.join(base_dir, data_dir))
            os.mkdir(os.path.join(base_dir, data_dir))

            rmtree(os.path.join(base_dir, subset_dir))
            os.mkdir(os.path.join(base_dir, subset_dir))

            # delete the existing channel_norm.feather and pixel_thresh.feather
            os.remove(channel_norm_path)
            os.remove(pixel_thresh_path)

    # create variable for storing 99.9% values
    quant_dat = pd.DataFrame()

    # find all the FOV files in the subsetted directory
    # NOTE: this handles the case where the data file was written, but not the subset file
    fovs_sub = io_utils.list_files(os.path.join(base_dir, subset_dir), substrs='.feather')

    # trim the .feather suffix from the fovs in the subsetted directory
    fovs_sub = io_utils.remove_file_extensions(fovs_sub)

    # define the list of FOVs for preprocessing
    # NOTE: if an existing FOV is already corrupted, future steps will discard it
    fovs_list = list(set(fovs).difference(set(fovs_sub)))

    # if there are no FOVs left to preprocess don't run function
    if len(fovs_list) == 0:
        print("There are no more FOVs to preprocess, skipping")
        return

    # if the process is only partially complete, inform the user of restart
    if len(fovs_list) < len(fovs):
        print("Restarting preprocessing from FOV %s, "
              "%d fovs left to process" % (fovs_list[0], len(fovs_list)))

    # check to make sure correct channels were specified
    check_for_modified_channels(tiff_dir=tiff_dir, test_fov=fovs[0], img_sub_folder=img_sub_folder,
                                channels=channels)

    # load existing channel_norm_path if exists, otherwise generate
    if not os.path.exists(channel_norm_path):
        # compute channel percentiles
        channel_norm_df = calculate_channel_percentiles(tiff_dir=tiff_dir,
                                                        fovs=fovs,
                                                        channels=channels,
                                                        img_sub_folder=img_sub_folder,
                                                        percentile=channel_percentile)
        # save output
        feather.write_dataframe(channel_norm_df, channel_norm_path, compression='uncompressed')
    else:
        # load previously generated output
        channel_norm_df = feather.read_dataframe(channel_norm_path)

    # load existing pixel_thresh_path if exists, otherwise generate
    if not os.path.exists(pixel_thresh_path):
        # compute pixel percentiles
        pixel_thresh_val = calculate_pixel_intensity_percentile(
            tiff_dir=tiff_dir, fovs=fovs, channels=channels,
            img_sub_folder=img_sub_folder, channel_percentiles=channel_norm_df
        )

        pixel_thresh_df = pd.DataFrame({'pixel_thresh_val': [pixel_thresh_val]})
        feather.write_dataframe(pixel_thresh_df, pixel_thresh_path, compression='uncompressed')
    else:
        pixel_thresh_df = feather.read_dataframe(pixel_thresh_path)
        pixel_thresh_val = pixel_thresh_df['pixel_thresh_val'].values[0]

    # define the partial function to iterate over
    fov_data_func = partial(
        preprocess_fov, base_dir, tiff_dir, data_dir, subset_dir,
        seg_dir, seg_suffix, img_sub_folder, is_mibitiff, channels, blur_factor,
        subset_proportion, pixel_thresh_val, seed, channel_norm_df
    )

    # define variable to keep track of number of fovs processed
    fovs_processed = 0

    # define the columns to drop for 99.9% normalization
    cols_to_drop = ['fov', 'row_index', 'column_index']

    # account for segmentation_label if seg_dir is set
    if seg_dir:
        cols_to_drop.append('segmentation_label')

    if multiprocess:
        # define the multiprocessing context
        with multiprocessing.get_context('spawn').Pool(batch_size) as fov_data_pool:
            # asynchronously generate and save the pixel matrices per FOV
            # NOTE: fov_data_pool should NOT operate on quant_dat since that is a shared resource
            for fov_batch in [fovs_list[i:(i + batch_size)]
                              for i in range(0, len(fovs_list), batch_size)]:
                fov_data_batch = fov_data_pool.map(fov_data_func, fov_batch)

                # compute the 99.9% quantile values for each FOV
                for pixel_mat_data in fov_data_batch:
                    # retrieve the FOV name, note that there will only be one per FOV DataFrame
                    fov = pixel_mat_data['fov'].unique()[0]

                    # drop the metadata columns and generate the 99.9% quantile values for the FOV
                    fov_full_pixel_data = pixel_mat_data.drop(columns=cols_to_drop)
                    quant_dat[fov] = fov_full_pixel_data.replace(
                        0, np.nan
                    ).quantile(q=0.999, axis=0)

                # update number of fovs processed
                fovs_processed += len(fov_batch)
                print("Processed %d fovs" % fovs_processed)
    else:
        for fov in fovs_list:
            pixel_mat_data = fov_data_func(fov)

            # drop the metadata columns and generate the 99.9% quantile values for the FOV
            fov_full_pixel_data = pixel_mat_data.drop(columns=cols_to_drop)
            quant_dat[fov] = fov_full_pixel_data.replace(0, np.nan).quantile(q=0.999, axis=0)

            # update number of fovs processed
            fovs_processed += 1

            # update every 10 FOVs, or at the very end
            if fovs_processed % 10 == 0 or fovs_processed == len(fovs_list):
                print("Processed %d fovs" % fovs_processed)

    # get mean 99.9% across all fovs for all markers
    mean_quant = pd.DataFrame(quant_dat.mean(axis=1))

    # save 99.9% normalization values
    feather.write_dataframe(mean_quant.T,
                            os.path.join(base_dir, norm_vals_name),
                            compression='uncompressed')


def find_fovs_missing_col(base_dir, data_dir, missing_col):
    """Identify FOV names in `data_dir` without `missing_col`

    Args:
        base_dir (str):
            The path to the data directories
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data
        missing_col (str):
            Name of the column to identify

    Returns:
        list:
            List of FOVs without `missing_col`
    """

    # define the main data path
    data_path = os.path.join(base_dir, data_dir)

    # define the temp data path
    temp_path = os.path.join(base_dir, data_dir + '_temp')

    # verify the data path exists
    io_utils.validate_paths(data_path)

    # if the temp path does not exist, either all the FOVs need to be run or none of them do
    if not os.path.exists(temp_path):
        # read in one of the FOV files in data_path
        fov_files = io_utils.list_files(data_path, substrs='.feather')

        # read in a sample FOV, do it this way to avoid potentially corrupted files
        # NOTE: this assumes only a few files get corrupted each run per MIBIAN runs
        # NOTE: handling of corrupted files gets propagated to respective R script
        i = 0
        while i < len(fov_files):
            try:
                fov_data = feather.read_dataframe(os.path.join(data_path, fov_files[i]))
            except (ArrowInvalid, OSError, IOError):
                i += 1
                continue
            break

        # if the missing_col is not found in fov_data, we need to run all the FOVs
        if missing_col not in fov_data.columns.values:
            # we will also make the temp directory to store the new files
            os.mkdir(temp_path)
            return io_utils.remove_file_extensions(fov_files)
        # otherwise, the column has already been assigned for this cohort, no need to run anything
        else:
            return []
    # if the temp path does exist, we have FOVs that need further processing
    else:
        # retrieve the FOV file names from both data_path and temp_path
        data_files = set(io_utils.list_files(data_path, substrs='.feather'))
        temp_files = set(io_utils.list_files(temp_path, substrs='.feather'))

        # get the difference between the two of these to determine which ones are valid
        leftover_files = list(data_files.difference(temp_files))

        return io_utils.remove_file_extensions(leftover_files)


def train_pixel_som(fovs, channels, base_dir,
                    subset_dir='pixel_mat_subsetted',
                    norm_vals_name='post_rowsum_chan_norm.feather',
                    som_weights_name='pixel_som_weights.feather', xdim=10, ydim=10,
                    lr_start=0.05, lr_end=0.01, num_passes=1):
    """Run the SOM training on the subsetted pixel data.

    Saves SOM weights to `base_dir/som_weights_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directories
        subset_dir (str):
            The name of the subsetted data directory
        norm_vals_name (str):
            The name of the file to store the 99.9% normalization values
        som_weights_name (str):
            The name of the file to save the SOM weights to
        xdim (int):
            The number of x nodes to use for the SOM
        ydim (int):
            The number of y nodes to use for the SOM
        lr_start (float):
            The start learning rate for the SOM, decays to `lr_end`
        lr_end (float):
            The end learning rate for the SOM, decays from `lr_start`
        num_passes (int):
            The number of training passes to make through the dataset

    Returns:
        cluster_helpers.PixelSOMCluster:
            The SOM cluster object containing the pixel SOM weights
    """

    # define the paths to the data
    subsetted_path = os.path.join(base_dir, subset_dir)
    norm_vals_path = os.path.join(base_dir, norm_vals_name)
    som_weights_path = os.path.join(base_dir, som_weights_name)

    # file path validation
    # NOTE: weights may or may not exist, that logic gets handled by PixelSOMCluster
    io_utils.validate_paths([subsetted_path, norm_vals_path])

    # verify that all provided fovs exist in the folder
    files = io_utils.list_files(subsetted_path, substrs='.feather')
    misc_utils.verify_in_list(provided_fovs=fovs,
                              subsetted_fovs=io_utils.remove_file_extensions(files))

    # verify that all the provided channels exist in subsetted data
    sample_sub = feather.read_dataframe(os.path.join(subsetted_path, files[0]))
    misc_utils.verify_in_list(provided_channels=channels,
                              subsetted_channels=sample_sub.columns.values)

    # define the pixel SOM cluster object
    pixel_pysom = cluster_helpers.PixelSOMCluster(
        subsetted_path, norm_vals_path, som_weights_path, channels,
        num_passes=num_passes, xdim=xdim, ydim=ydim, lr_start=lr_start, lr_end=lr_end
    )

    # train the SOM weights
    # NOTE: seed has to be set in cyFlowSOM.pyx, done by passing flag in PixieSOMCluster
    print("Training SOM")
    pixel_pysom.train_som()

    return pixel_pysom


def run_pixel_som_assignment(pixel_data_path, pixel_pysom_obj, fov):
    """Helper function to assign pixel SOM cluster labels

    Args:
        pixel_data_path (str):
            The path to the pixel data directory
        pixel_pysom_obj (ark.phenotyping.cluster_helpers.PixieConsensusCluster):
            The pixel SOM cluster object
        fov (str):
            The name of the FOV to process

    Returns:
        tuple (str, int):
            The name of the FOV as well as the return code
    """

    # get the path to the fov
    fov_path = os.path.join(pixel_data_path, fov + '.feather')

    # read in the fov data with SOM labels
    try:
        fov_data = feather.read_dataframe(fov_path)
    # this indicates this fov file is corrupted
    except (ArrowInvalid, OSError, IOError):
        return fov, 1

    # assign the SOM labels to fov_data
    fov_data = pixel_pysom_obj.assign_som_clusters(fov_data)

    # resave the data with the SOM cluster labels assigned
    temp_path = os.path.join(pixel_data_path + '_temp', fov + '.feather')
    feather.write_dataframe(fov_data, temp_path, compression='uncompressed')

    return fov, 0


def cluster_pixels(fovs, channels, base_dir, pixel_pysom, data_dir='pixel_mat_data',
                   multiprocess=False, batch_size=5):
    """Uses trained SOM weights to assign cluster labels on full pixel data.

    Saves data with cluster labels to `data_dir`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        pixel_pysom (cluster_helpers.PixelSOMCluster):
            The SOM cluster object containing the pixel SOM weights
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data
        multiprocess (bool):
            Whether to use multiprocessing or not
        batch_size (int):
            The number of FOVs to process in parallel, ignored if `multiprocess` is `False`
    """

    # define the paths to the data
    data_path = os.path.join(base_dir, data_dir)

    # path validation
    io_utils.validate_paths([data_path])

    # raise error if weights haven't been assigned to pixel_pysom
    if pixel_pysom.weights is None:
        raise ValueError("Using untrained pixel_pysom object, please invoke train_som first")

    # verify that all provided fovs exist in the folder
    # NOTE: remove the channel and pixel normalization files as those are not pixel data
    data_files = io_utils.list_files(data_path, substrs='.feather')
    misc_utils.verify_in_list(provided_fovs=fovs,
                              subsetted_fovs=io_utils.remove_file_extensions(data_files))

    # this will prevent reading in a corrupted sample_fov
    i = 0
    while i < len(data_files):
        try:
            sample_fov = feather.read_dataframe(os.path.join(base_dir, data_dir, data_files[i]))
        except (ArrowInvalid, OSError, IOError):
            i += 1
            continue
        break

    # for verification purposes, drop the metadata columns
    cols_to_drop = ['fov', 'row_index', 'column_index']
    for col in ['segmentation_label', 'pixel_som_cluster',
                'pixel_meta_cluster', 'pixel_meta_cluster_rename']:
        if col in sample_fov.columns.values:
            cols_to_drop.append(col)

    sample_fov = sample_fov.drop(
        columns=cols_to_drop
    )
    misc_utils.verify_same_elements(
        enforce_order=True,
        norm_vals_columns=pixel_pysom.norm_data.columns.values,
        pixel_data_columns=sample_fov.columns.values
    )

    # ensure the SOM weights columns are valid indexes
    misc_utils.verify_same_elements(
        enforce_order=True,
        pixel_som_weights_columns=pixel_pysom.weights.columns.values,
        pixel_data_columns=sample_fov.columns.values
    )

    # only assign SOM clusters to FOVs that don't already have them
    fovs_list = find_fovs_missing_col(base_dir, data_dir, 'pixel_som_cluster')

    # if there are no FOVs left without SOM labels don't run function
    if len(fovs_list) == 0:
        print("There are no more FOVs to assign SOM labels to, skipping")
        return

    # if SOM cluster labeling is only partially complete, inform the user of restart
    if len(fovs_list) < len(fovs):
        print("Restarting SOM label assignment from fov %s, "
              "%d fovs left to process" % (fovs_list[0], len(fovs_list)))

    # define variable to keep track of number of fovs processed
    fovs_processed = 0

    # define the partial function to iterate over
    fov_data_func = partial(
        run_pixel_som_assignment, data_path, pixel_pysom
    )

    # use the som weights to assign SOM cluster values to data in data_dir
    print("Mapping pixel data to SOM cluster labels")

    if multiprocess:
        with multiprocessing.get_context('spawn').Pool(batch_size) as fov_data_pool:
            for fov_batch in [fovs_list[i:(i + batch_size)]
                              for i in range(0, len(fovs_list), batch_size)]:
                fov_statuses = fov_data_pool.map(fov_data_func, fov_batch)

                for fs in fov_statuses:
                    if fs[1] == 1:
                        print("The data for FOV %s has been corrupted, skipping" % fs[0])
                        fovs_processed -= 1

                # update number of fovs processed
                fovs_processed += len(fov_batch)

                print("Processed %d fovs" % fovs_processed)
    else:
        for fov in fovs_list:
            fov_status = fov_data_func(fov)

            if fov_status[1] == 1:
                print("The data for FOV %s has been corrupted, skipping" % fov_status[0])
                fovs_processed -= 1

            # update number of fovs processed
            fovs_processed += 1

            # update every 10 FOVs, or at the very end
            if fovs_processed % 10 == 0 or fovs_processed == len(fovs_list):
                print("Processed %d fovs" % fovs_processed)

    # remove the data directory and rename the temp directory to the data directory
    rmtree(data_path)
    os.rename(data_path + '_temp', data_path)


def generate_som_avg_files(fovs, channels, base_dir, pixel_pysom, data_dir='pixel_data_dir',
                           pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                           num_fovs_subset=100, seed=42):
    """Computes and saves the average channel expression across pixel SOM clusters.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        pixel_pysom (cluster_helpers.PixelSOMCluster):
            The SOM cluster object containing the pixel SOM weights
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data
        pc_chan_avg_som_cluster_name (str):
            The name of the file to save the average channel expression across all SOM clusters
        num_fovs_subset (int):
            The number of FOVs to subset on for SOM cluster channel averaging
        seed (int):
            The random seed to set for subsetting FOVs
    """

    # define the paths to the data
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)

    # raise error if weights haven't been assigned to pixel_pysom
    if pixel_pysom.weights is None:
        raise ValueError("Using untrained pixel_pysom object, please invoke train_som first")

    # if the channel SOM average file already exists, skip
    if os.path.exists(som_cluster_avg_path):
        print("Already generated SOM cluster channel average file, skipping")
        return

    # compute average channel expression for each pixel SOM cluster
    # and the number of pixels per SOM cluster
    print("Computing average channel expression across pixel SOM clusters")
    pixel_channel_avg_som_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_som_cluster',
        pixel_pysom.weights.shape[0],
        data_dir,
        num_fovs_subset=num_fovs_subset,
        seed=seed,
        keep_count=True
    )

    # save pixel_channel_avg_som_cluster
    pixel_channel_avg_som_cluster.to_csv(
        som_cluster_avg_path,
        index=False
    )


def run_pixel_consensus_assignment(pixel_data_path, pixel_cc_obj, fov):
    """Helper function to assign pixel consensus clusters

    Args:
        pixel_data_path (str):
            The path to the pixel data directory
        pixel_cc_obj (ark.phenotyping.cluster_helpers.PixieConsensusCluster):
            The pixel consensus cluster object
        fov (str):
            The name of the FOV to process

    Returns:
        tuple (str, int):
            The name of the FOV as well as the return code
    """

    # get the path to the fov
    fov_path = os.path.join(pixel_data_path, fov + '.feather')

    # read in the fov data with SOM labels
    try:
        fov_data = feather.read_dataframe(fov_path)
    # this indicates this fov file is corrupted
    except (ArrowInvalid, OSError, IOError):
        return fov, 1

    # assign the consensus labels to fov_data
    fov_data = pixel_cc_obj.assign_consensus_labels(fov_data)

    # resave the data with the meta cluster labels assigned
    temp_path = os.path.join(pixel_data_path + '_temp', fov + '.feather')
    feather.write_dataframe(fov_data, temp_path, compression='uncompressed')

    return fov, 0


def pixel_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                            data_dir='pixel_mat_data',
                            pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                            multiprocess=False, batch_size=5, seed=42):
    """Run consensus clustering algorithm on pixel-level summed data across channels
    Saves data with consensus cluster labels to `data_dir`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        max_k (int):
            The number of consensus clusters
        cap (int):
            z-score cap to use when hierarchical clustering
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data.
            This data should also have the SOM cluster labels appended from `cluster_pixels`.
        pc_chan_avg_som_cluster_name (str):
            Name of file to save the channel-averaged results across all SOM clusters to
        multiprocess (bool):
            Whether to use multiprocessing or not
        batch_size (int):
            The number of FOVs to process in parallel, ignored if `multiprocess` is `False`
        seed (int):
            The random seed to set for consensus clustering

    Returns:
        cluster_helpers.PixieConsensusCluster:
            The consensus cluster object containing the SOM to meta mapping
    """

    # define the paths to the data
    pixel_data_path = os.path.join(base_dir, data_dir)
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)

    # path validation
    io_utils.validate_paths([pixel_data_path, som_cluster_avg_path])

    # only assign meta clusters to FOVs that don't already have them
    fovs_list = find_fovs_missing_col(base_dir, data_dir, 'pixel_meta_cluster')

    # if there are no FOVs left without meta labels don't run function
    if len(fovs_list) == 0:
        print("There are no more FOVs to assign meta labels to, skipping")
        return

    # if meta cluster labeling is only partially complete, inform the user of restart
    if len(fovs_list) < len(fovs):
        print("Restarting meta cluster label assignment from fov %s, "
              "%d fovs left to process" % (fovs_list[0], len(fovs_list)))

    # consensus clustering setup
    pixel_cc = cluster_helpers.PixieConsensusCluster(
        'pixel', som_cluster_avg_path, channels, max_k=max_k, cap=cap
    )

    # z-score and cap the data
    print("z-score scaling and capping data")
    pixel_cc.scale_data()

    # set random seed for consensus clustering
    np.random.seed(seed)

    # run consensus clustering
    print("Running consensus clustering")
    pixel_cc.run_consensus_clustering()

    # generate the the som to meta cluster map
    pixel_cc.generate_som_to_meta_map()

    # define variable to keep track of number of fovs processed
    fovs_processed = 0

    # define the partial function to iterate over
    fov_data_func = partial(
        run_pixel_consensus_assignment, pixel_data_path, pixel_cc
    )

    # use the som to meta mapping to assign meta cluster values to data in data_path
    print("Mapping pixel data to consensus cluster labels")

    # TODO: this multiprocess logic will be duplicated in several places, should be own function
    if multiprocess:
        with multiprocessing.get_context('spawn').Pool(batch_size) as fov_data_pool:
            for fov_batch in [fovs_list[i:(i + batch_size)]
                              for i in range(0, len(fovs_list), batch_size)]:
                fov_statuses = fov_data_pool.map(fov_data_func, fov_batch)

                for fs in fov_statuses:
                    if fs[1] == 1:
                        print("The data for FOV %s has been corrupted, skipping" % fs[0])
                        fovs_processed -= 1

                # update number of fovs processed
                fovs_processed += len(fov_batch)

                print("Processed %d fovs" % fovs_processed)
    else:
        for fov in fovs_list:
            fov_status = fov_data_func(fov)

            if fov_status[1] == 1:
                print("The data for FOV %s has been corrupted, skipping" % fov_status[0])
                fovs_processed -= 1

            # update number of fovs processed
            fovs_processed += 1

            # update every 10 FOVs, or at the very end
            if fovs_processed % 10 == 0 or fovs_processed == len(fovs_list):
                print("Processed %d fovs" % fovs_processed)

    # remove the data directory and rename the temp directory to the data directory
    rmtree(pixel_data_path)
    os.rename(pixel_data_path + '_temp', pixel_data_path)

    return pixel_cc


def generate_meta_avg_files(fovs, channels, base_dir, pixel_cc, data_dir='pixel_mat_data',
                            pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                            pc_chan_avg_meta_cluster_name='pixel_channel_avg_meta_cluster.csv',
                            num_fovs_subset=100, seed=42):
    """Computes and saves the average channel expression across pixel meta clusters.
    Assigns meta cluster labels to the data stored in `pc_chan_avg_som_cluster_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        pixel_cc (cluster_helpers.PixieConsensusCluster):
            The consensus cluster object containing the SOM to meta mapping
        data_dir (str):
            Name of the directory which contains the full preprocessed pixel data.
            This data should also have the SOM cluster labels appended from `cluster_pixels`.
        pc_chan_avg_som_cluster_name (str):
            Name of file to save the channel-averaged results across all SOM clusters to
        pc_chan_avg_meta_cluster_name (str):
            Name of file to save the channel-averaged results across all meta clusters to
        num_fovs_subset (float):
            The number of FOVs to subset on for meta cluster channel averaging
        seed (int):
            The random seed to use for subsetting FOVs
    """

    # define the paths to the data
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)
    meta_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_meta_cluster_name)

    # path validation
    io_utils.validate_paths([som_cluster_avg_path])

    # if the channel meta average file already exists, skip
    if os.path.exists(meta_cluster_avg_path):
        print("Already generated meta cluster channel average file, skipping")
        return

    # compute average channel expression for each pixel meta cluster
    # and the number of pixels per meta cluster
    print("Computing average channel expression across pixel meta clusters")
    pixel_channel_avg_meta_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_meta_cluster',
        pixel_cc.max_k,
        data_dir,
        num_fovs_subset=num_fovs_subset,
        seed=seed,
        keep_count=True
    )

    # save pixel_channel_avg_meta_cluster
    pixel_channel_avg_meta_cluster.to_csv(
        meta_cluster_avg_path,
        index=False
    )

    # merge metacluster assignments in
    print("Mapping meta cluster values onto average channel expression across pixel SOM clusters")
    pixel_channel_avg_som_cluster = pd.read_csv(som_cluster_avg_path)
    pixel_channel_avg_som_cluster = pd.merge_asof(
        pixel_channel_avg_som_cluster, pixel_cc.mapping, on='pixel_som_cluster'
    )

    # resave channel-averaged results across all pixel SOM clusters with metacluster assignments
    pixel_channel_avg_som_cluster.to_csv(
        som_cluster_avg_path,
        index=False
    )


def update_pixel_meta_labels(pixel_data_path, pixel_remapped_dict,
                             pixel_renamed_meta_dict, fov):
    """Helper function to reassign meta cluster names based on remapping scheme to a FOV

    Args:
        pixel_data_path (str):
            The path to the pixel data drectory
        pixel_remapped_dict (dict):
            The mapping from pixel SOM cluster to pixel meta cluster label (not renamed)
        pixel_renamed_meta_dict (dict):
            The mapping from pixel meta cluster label to renamed pixel meta cluster name
        fov (str):
            The name of the FOV to process

    Returns:
        tuple (str, int):
            The name of the FOV as well as the return code
    """

    # get the path to the fov
    fov_path = os.path.join(pixel_data_path, fov + '.feather')

    # read in the fov data with SOM and meta cluster labels
    try:
        fov_data = feather.read_dataframe(fov_path)
    # this indicates this fov file is corrupted
    except (ArrowInvalid, OSError, IOError):
        return fov, 1

    # ensure that no SOM clusters are missing from the mapping
    misc_utils.verify_in_list(
        fov_som_labels=fov_data['pixel_som_cluster'].unique(),
        som_labels_in_mapping=list(pixel_remapped_dict.keys())
    )

    # assign the new meta cluster labels
    fov_data['pixel_meta_cluster'] = fov_data['pixel_som_cluster'].map(
        pixel_remapped_dict
    )

    # assign the renamed meta cluster names
    fov_data['pixel_meta_cluster_rename'] = fov_data['pixel_meta_cluster'].map(
        pixel_renamed_meta_dict
    )

    # resave the data with the new meta cluster labels
    temp_path = os.path.join(pixel_data_path + '_temp', fov + '.feather')
    feather.write_dataframe(fov_data, temp_path, compression='uncompressed')

    return fov, 0


def apply_pixel_meta_cluster_remapping(fovs, channels, base_dir,
                                       pixel_data_dir,
                                       pixel_remapped_name,
                                       multiprocess=False, batch_size=5):
    """Apply the meta cluster remapping to the data in `pixel_data_dir`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_data_dir (str):
            Name of directory with the full pixel data.
            This data should also have the SOM cluster labels appended from `cluster_pixels`
            and the meta cluster labels appended from `pixel_consensus_cluster`.
        pixel_remapped_name (str):
            Name of the file containing the pixel SOM clusters to their remapped meta clusters
        multiprocess (bool):
            Whether to use multiprocessing or not
        batch_size (int):
            The number of FOVs to process in parallel
    """

    # define the data paths
    pixel_data_path = os.path.join(base_dir, pixel_data_dir)
    pixel_remapped_path = os.path.join(base_dir, pixel_remapped_name)

    # file path validation
    io_utils.validate_paths([pixel_data_path, pixel_remapped_path])

    # read in the remapping
    pixel_remapped_data = pd.read_csv(pixel_remapped_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapped_data_cols=pixel_remapped_data.columns.values,
        required_cols=['cluster', 'metacluster', 'mc_name']
    )

    # rename columns in pixel_remapped_data so it plays better with the existing
    # pixel_som_cluster and pixel_meta_cluster
    pixel_remapped_data = pixel_remapped_data.rename(
        {
            'cluster': 'pixel_som_cluster',
            'metacluster': 'pixel_meta_cluster',
            'mc_name': 'pixel_meta_cluster_rename'
        },
        axis=1
    )

    # create the mapping from pixel SOM to pixel meta cluster
    pixel_remapped_dict = dict(
        pixel_remapped_data[
            ['pixel_som_cluster', 'pixel_meta_cluster']
        ].values
    )

    # create the mapping from pixel meta cluster to renamed pixel meta cluster
    pixel_renamed_meta_dict = dict(
        pixel_remapped_data[
            ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ].drop_duplicates().values
    )

    # define the partial function to iterate over
    fov_data_func = partial(
        update_pixel_meta_labels, pixel_data_path,
        pixel_remapped_dict, pixel_renamed_meta_dict
    )

    # if it doesn't already exist, create a temporary directory to write the data
    if not os.path.exists(pixel_data_path + '_temp'):
        os.mkdir(pixel_data_path + '_temp')
        fov_list = fovs
    # otherwise, re-run only for unprocessed FOVs
    else:
        # NOTE: this will only test the else statement in find_fovs_missing_col
        fov_list = find_fovs_missing_col(base_dir, pixel_data_dir, 'pixel_meta_cluster_rename')
        print("Restarting meta cluster remapping assignment from %s, "
              "%d fovs left to process" % (fov_list[0], len(fov_list)))

    # define variable to keep track of number of fovs processed
    fovs_processed = 0

    print("Using re-mapping scheme to re-label pixel meta clusters")
    if multiprocess:
        # define the multiprocessing context
        with multiprocessing.get_context('spawn').Pool(batch_size) as fov_data_pool:
            # asynchronously generate and save the pixel matrices per FOV
            for fov_batch in [fov_list[i:(i + batch_size)]
                              for i in range(0, len(fov_list), batch_size)]:
                fov_statuses = fov_data_pool.map(fov_data_func, fov_batch)

                for fs in fov_statuses:
                    if fs[1] == 1:
                        print("The data for FOV %s has been corrupted, skipping" % fs[0])
                        fovs_processed -= 1

                # update number of fovs processed
                fovs_processed += len(fov_batch)

                print("Processed %d fovs" % fovs_processed)
    else:
        for fov in fov_list:
            fov_status = fov_data_func(fov)

            if fov_status[1] == 1:
                print("The data for FOV %s has been corrupted, skipping" % fov_status[0])
                fovs_processed -= 1

            # update number of fovs processed
            fovs_processed += 1

            # update every 10 FOVs, or at the very end
            if fovs_processed % 10 == 0 or fovs_processed == len(fov_list):
                print("Processed %d fovs" % fovs_processed)

    # remove the data directory and rename the temp directory to the data directory
    rmtree(pixel_data_path)
    os.rename(pixel_data_path + '_temp', pixel_data_path)


def generate_remap_avg_files(fovs, channels, base_dir, pixel_data_dir, pixel_remapped_name,
                             pc_chan_avg_som_cluster_name, pc_chan_avg_meta_cluster_name,
                             num_fovs_subset=100, seed=42):
    """Resaves the re-mapped consensus data to `pixel_data_dir` and re-runs the
    average channel expression per pixel meta cluster computation.

    Re-maps the pixel SOM clusters to meta clusters in `pc_chan_avg_som_cluster_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_data_dir (str):
            Name of directory with the full pixel data.
            This data should also have the SOM cluster labels appended from `cluster_pixels`
            and the meta cluster labels appended from `pixel_consensus_cluster`.
        pixel_remapped_name (str):
            Name of the file containing the pixel SOM clusters to their remapped meta clusters
        pc_chan_avg_som_cluster_name (str):
            Name of the file containing the channel-averaged results across all SOM clusters
        pc_chan_avg_meta_cluster_name (str):
            Name of the file containing the channel-averaged results across all meta clusters
        num_fovs_subset (float):
            The number of FOVs to subset on for meta cluster channel averaging
        seed (int):
            The random seed to use for subsetting FOVs
    """

    # define the data paths
    pixel_remapped_path = os.path.join(base_dir, pixel_remapped_name)
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)
    meta_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_meta_cluster_name)

    # file path validation
    io_utils.validate_paths([pixel_remapped_path, som_cluster_avg_path, meta_cluster_avg_path])

    # read in the unique meta clusters defined in pixel_remapped_path
    new_meta_clusters = pd.read_csv(pixel_remapped_path)['metacluster'].unique()

    # read in the remapping
    # TODO: define a separate function for this duplicated logic, OOP will help greatly (soon...)
    pixel_remapped_data = pd.read_csv(pixel_remapped_path)

    # rename columns in pixel_remapped_data so it plays better with the existing
    # pixel_som_cluster and pixel_meta_cluster
    pixel_remapped_data = pixel_remapped_data.rename(
        {
            'cluster': 'pixel_som_cluster',
            'metacluster': 'pixel_meta_cluster',
            'mc_name': 'pixel_meta_cluster_rename'
        },
        axis=1
    )

    # create the mapping from pixel SOM to pixel meta cluster
    pixel_remapped_dict = dict(
        pixel_remapped_data[
            ['pixel_som_cluster', 'pixel_meta_cluster']
        ].values
    )

    # create the mapping from pixel meta cluster to renamed pixel meta cluster
    pixel_renamed_meta_dict = dict(
        pixel_remapped_data[
            ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ].drop_duplicates().values
    )

    # re-compute average channel expression for each pixel meta cluster
    # and the number of pixels per meta cluster, add renamed meta cluster column in
    print("Re-computing average channel expression across pixel meta clusters")
    pixel_channel_avg_meta_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_meta_cluster',
        len(pixel_remapped_data['pixel_meta_cluster'].unique()),
        pixel_data_dir,
        num_fovs_subset=num_fovs_subset,
        seed=seed,
        keep_count=True
    )
    pixel_channel_avg_meta_cluster['pixel_meta_cluster_rename'] = \
        pixel_channel_avg_meta_cluster['pixel_meta_cluster'].map(pixel_renamed_meta_dict)

    # re-save the pixel channel average meta cluster table
    pixel_channel_avg_meta_cluster.to_csv(meta_cluster_avg_path, index=False)

    # re-assign pixel meta cluster labels back to the pixel channel average som cluster table
    pixel_channel_avg_som_cluster = pd.read_csv(som_cluster_avg_path)

    print("Re-assigning meta cluster column in pixel SOM cluster average channel expression table")
    pixel_channel_avg_som_cluster['pixel_meta_cluster'] = \
        pixel_channel_avg_som_cluster['pixel_som_cluster'].map(pixel_remapped_dict)

    pixel_channel_avg_som_cluster['pixel_meta_cluster_rename'] = \
        pixel_channel_avg_som_cluster['pixel_meta_cluster'].map(pixel_renamed_meta_dict)

    # re-save the pixel channel average som cluster table
    pixel_channel_avg_som_cluster.to_csv(som_cluster_avg_path, index=False)
