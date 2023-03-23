import multiprocessing
import os
import random
from shutil import rmtree

import feather
from functools import partial
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage.io import imread
from alpineer import io_utils, load_utils, misc_utils

from ark.phenotyping import pixel_cluster_utils

multiprocessing.set_start_method('spawn', force=True)


def create_fov_pixel_data(fov, channels, img_data, seg_labels,
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

    # remove any rows with channels that sum to zero prior to normalization
    pixel_mat = pixel_mat.loc[(pixel_mat[channels] != 0).any(axis=1), :].reset_index(drop=True)

    # normalize the row sums of pixel mat
    pixel_mat = pixel_cluster_utils.normalize_rows(pixel_mat, channels, seg_labels is not None)

    # subset the pixel matrix for training
    pixel_mat_subset = pixel_mat.sample(frac=subset_proportion)

    return pixel_mat, pixel_mat_subset


def preprocess_fov(base_dir, tiff_dir, data_dir, subset_dir, seg_dir, seg_suffix,
                   img_sub_folder, is_mibitiff, channels, blur_factor,
                   subset_proportion, seed, fov):
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
        seed (int):
            The random seed to set for subsetting
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

    # set seed for subsetting
    np.random.seed(seed)

    # create the full and subsetted fov matrices
    pixel_mat, pixel_mat_subset = create_fov_pixel_data(
        fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
        blur_factor=blur_factor,
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

    # define the partial function to iterate over
    fov_data_func = partial(
        preprocess_fov, base_dir, tiff_dir, data_dir, subset_dir,
        seg_dir, seg_suffix, img_sub_folder, is_mibitiff, channels, blur_factor,
        subset_proportion, seed
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
