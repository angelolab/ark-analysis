import os
import random
import warnings
from typing import List

import feather
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from pyarrow.lib import ArrowInvalid
from skimage.io import imread

from alpineer import image_utils, io_utils, load_utils, misc_utils


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
            image_utils.save_image(
                os.path.join(tiff_dir, fov, img_sub_folder, chan + '_smoothed.tiff'),
                chan_out
            )


def filter_with_nuclear_mask(fovs: List, tiff_dir: str, seg_dir: str, channel: str,
                             nuc_seg_suffix: str = "_nuclear.tiff", img_sub_folder: str = None,
                             exclude: bool = True):
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
        nuc_seg_suffix (str):
            The suffix for the nuclear channel.
            (i.e. for "fov1", a suffix of "_nuclear.tiff" would make a file named
            "fov1_nuclear.tiff")
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
        seg_img_name: str = f"{fov}{nuc_seg_suffix}"
        seg_img = imread(os.path.join(seg_dir, seg_img_name))[0, ...]

        # mask out the nucleus
        if exclude:
            suffix = "_nuc_exclude.tiff"
            seg_mask = seg_img > 0
        # mask out the membrane
        else:
            suffix = "_nuc_include.tiff"
            seg_mask = seg_img == 0

        # filter out the nucleus or membrane depending on exclude parameter
        img[seg_mask] = 0

        # save filtered image
        image_utils.save_image(os.path.join(tiff_dir, fov, img_sub_folder, channel + suffix), img)


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
