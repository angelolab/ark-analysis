import multiprocessing
import os
from functools import partial
from shutil import rmtree

import feather
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers
from ark.phenotyping import pixel_cluster_utils

multiprocessing.set_start_method('spawn', force=True)


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

    # find all the FOV files in the full data and subsetted directories
    # NOTE: this handles the case where the data file was written, but not the subset file
    fovs_sub = io_utils.list_files(os.path.join(base_dir, subset_dir), substrs='.feather')
    fovs_data = io_utils.list_files(os.path.join(base_dir, data_dir), substrs='.feather')

    # intersect the two fovs lists together (if a FOV appears in one but not the other, regenerate)
    fovs_full = list(set(fovs_sub).intersection(fovs_data))

    # trim the .feather suffix from the fovs in the subsetted directory
    fovs_full = io_utils.remove_file_extensions(fovs_full)

    # define the list of FOVs for preprocessing
    # NOTE: if an existing FOV is already corrupted, future steps will discard it
    fovs_list = list(set(fovs).difference(set(fovs_full)))

    # if there are no FOVs left to preprocess don't run function
    if len(fovs_list) == 0:
        print("There are no more FOVs to preprocess, skipping")
        return

    # if the process is only partially complete, inform the user of restart
    if len(fovs_list) < len(fovs):
        print("Restarting preprocessing from FOV %s, "
              "%d fovs left to process" % (fovs_list[0], len(fovs_list)))

    # check to make sure correct channels were specified
    pixel_cluster_utils.check_for_modified_channels(
        tiff_dir=tiff_dir, 
        test_fov=fovs[0],
        img_sub_folder=img_sub_folder,
        channels=channels
    )

    # load existing channel_norm_path if exists, otherwise generate
    if not os.path.exists(channel_norm_path):
        # compute channel percentiles
        channel_norm_df = pixel_cluster_utils.calculate_channel_percentiles(
            tiff_dir=tiff_dir,
            fovs=fovs,
            channels=channels,
            img_sub_folder=img_sub_folder,
            percentile=channel_percentile
        )
        # save output
        feather.write_dataframe(channel_norm_df, channel_norm_path, compression='uncompressed')
    else:
        # load previously generated output
        channel_norm_df = feather.read_dataframe(channel_norm_path)

    # load existing pixel_thresh_path if exists, otherwise generate
    if not os.path.exists(pixel_thresh_path):
        # compute pixel percentiles
        pixel_thresh_val = pixel_cluster_utils.calculate_pixel_intensity_percentile(
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
        pixel_cluster_utils.preprocess_fov, base_dir, tiff_dir, data_dir, subset_dir,
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


def train_pixel_som(fovs, channels, base_dir,
                    subset_dir='pixel_mat_subsetted',
                    norm_vals_name='post_rowsum_chan_norm.feather',
                    som_weights_name='pixel_som_weights.feather', xdim=10, ydim=10,
                    lr_start=0.05, lr_end=0.01, num_passes=1, seed=42,
                    overwrite=False):
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
        seed (int):
            The random seed to use for training the SOM
        overwrite (bool):
            If set, force retrains the SOM and overwrites the weights

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
        subsetted_path, norm_vals_path, som_weights_path, fovs, channels,
        num_passes=num_passes, xdim=xdim, ydim=ydim, lr_start=lr_start, lr_end=lr_end,
        seed=seed
    )

    # train the SOM weights
    # NOTE: seed has to be set in cyFlowSOM.pyx, done by passing flag in PixieSOMCluster
    print("Training SOM")
    pixel_pysom.train_som(overwrite=overwrite)

    return pixel_pysom


def cluster_pixels(fovs, channels, base_dir, pixel_pysom, data_dir='pixel_mat_data',
                   multiprocess=False, batch_size=5, overwrite=False):
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
        overwrite (bool):
            If set, force overwrite the SOM labels in all the FOVs
    """

    # define the paths to the data
    data_path = os.path.join(base_dir, data_dir)

    # path validation
    io_utils.validate_paths([data_path])

    # raise error if weights haven't been assigned to pixel_pysom
    if pixel_pysom.weights is None:
        raise ValueError("Using untrained pixel_pysom object, please invoke train_pixel_som first")

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

    # if overwrite flag set, run on all FOVs in data_dir
    if overwrite:
        print('Overwrite flag set, reassigning SOM cluster labels to all FOVs')
        os.mkdir(data_path + '_temp')
        fovs_list = io_utils.remove_file_extensions(
            io_utils.list_files(data_path, substrs='.feather')
        )
    # otherwise, only assign SOM clusters to FOVs that don't already have them
    else:
        fovs_list = pixel_cluster_utils.find_fovs_missing_col(
            base_dir, data_dir, 'pixel_som_cluster'
        )

    # make sure fovs_list only contain fovs that exist in the master fovs list specified
    fovs_list = list(set(fovs_list).intersection(fovs))

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
        pixel_cluster_utils.run_pixel_som_assignment, data_path, pixel_pysom
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
                           num_fovs_subset=100, seed=42, overwrite=False):
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
        overwrite (bool):
            If set, force overwrite the existing average channel expression file if it exists
    """

    # define the paths to the data
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)

    # raise error if weights haven't been assigned to pixel_pysom
    if pixel_pysom.weights is None:
        raise ValueError("Using untrained pixel_pysom object, please invoke train_som first")

    # if the channel SOM average file already exists and the overwrite flag isn't set, skip
    if os.path.exists(som_cluster_avg_path):
        if not overwrite:
            print("Already generated SOM cluster channel average file, skipping")
            return

        print("Overwrite flag set, regenerating SOM cluster channel average file")

    # compute average channel expression for each pixel SOM cluster
    # and the number of pixels per SOM cluster
    print("Computing average channel expression across pixel SOM clusters")
    pixel_channel_avg_som_cluster = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
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
