import cv2
from numba import njit
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import xarray as xr

import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils

import timeit


def compute_nonzero_mean_intensity(image_data):
    """Compute the nonzero mean of each channel per fov

    Args:
        image_data (xarray.DataArray):
            the image data with fov and channel dimensions

    Returns:
        numpy.ndarray:
            Matrix indicating the nonzero mean intensity of each channel per fov
    """

    # define the nonzero_mean_intensity array
    nonzero_mean_intensity = np.zeros((image_data.shape[0], image_data.shape[3]))

    # iterate over each fov and channel
    for i in np.arange(image_data.shape[0]):
        for j in np.arange(image_data.shape[3]):
            # extract the data for the fov and channel
            image_data_np = image_data[i, :, :, j].values

            # take just the non-zero pixels
            image_data_nonzero = image_data_np[image_data_np != 0]

            # take the mean of the non-zero pixels and assign to (fov, channel) in array
            # unless there are no non-zero pixels, in which case default to np.nan
            if len(image_data_nonzero) > 0:
                nonzero_mean_intensity[i, j] = image_data_nonzero.mean()
            else:
                nonzero_mean_intensity[i, j] = np.nan

    return nonzero_mean_intensity


def compute_total_intensity(image_data):
    """Compute the total intensity of each channel per fov

    Args:
        image_data (xarray.DataArray):
            the image data with fov and channel dimensions

    Returns:
        pandas.DataFrame:
            Matrix indicating the nonzero mean intensity of each channel per fov
    """

    # a broadcasted approach is faster for sum
    return image_data.values.sum(axis=(1, 2))


def compute_99_9_intensity(image_data):
    """Compute the 99.9% intensity value of each channel per fov

    Args:
        image_data (xarray.DataArray):
            the image data with fov and channel dimensions

    Returns:
        pandas.DataFrame:
            Matrix indicating the 99.9% intensity of each channel per fov
    """

    # define the 99.9% intensity array
    intensity_99_9 = np.zeros((image_data.shape[0], image_data.shape[3]))

    # iterate over each fov and channel
    for i in np.arange(image_data.shape[0]):
        for j in np.arange(image_data.shape[3]):
            # extract the data for the fov and channel
            image_data_np = image_data[i, :, :, j].values

            # take the 99.9% value of the data and assign
            intensity_99_9[i, j] = np.percentile(image_data_np, q=99.9)

    return intensity_99_9


def compute_qc_metrics(tiff_dir, img_sub_folder="TIFs", is_mibitiff=False,
                       fovs=None, chans=None, batch_size=5, gaussian_blur=False,
                       blur_factor=1, dtype=np.float32):
    """Compute the QC metric matrices

    Args:
        tiff_dir (str):
            the name of the directory which contains the single_channel_inputs
        img_sub_folder (str):
            the name of the folder where the TIF images are located
            ignored if is_mibitiff is True
        is_mibitiff (bool):
            a flag to indicate whether or not the base images are MIBItiffs
        fovs (list):
            a list of fovs we wish to analyze, if None will default to all fovs
        chans (list):
            a list of channels we wish to subset on, if None will default to all channels
        batch_size (int):
            how large we want each of the batches of fovs to be when computing, adjust as
            necessary for speed and memory considerations
        gaussian_blur (bool):
            whether or not to add Gaussian blurring
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
            ignored if gaussian_blur set to False
        dtype (str/type):
            data type of base images

    Returns:
        dict:
            A mapping between each QC metric name and their respective DataFrames
    """

    # if no fovs are specified, then load all the fovs
    if fovs is None:
        if is_mibitiff:
            fovs = io_utils.list_files(tiff_dir, substrs=['.tif', '.tiff'])
        else:
            fovs = io_utils.list_folders(tiff_dir)

    # drop file extensions
    fovs = io_utils.remove_file_extensions(fovs)

    # get full filenames from given fovs
    filenames = io_utils.list_files(tiff_dir, substrs=fovs, exact_match=True)

    # sort the fovs and filenames
    fovs.sort()
    filenames.sort()

    # define the number of fovs specified
    cohort_len = len(fovs)

    # create the DataFrames to store the processed data
    df_nonzero_mean = pd.DataFrame()
    df_total_intensity = pd.DataFrame()
    df_99_9_intensity = pd.DataFrame()

    # define number of fovs processed (for printing out update to user)
    fovs_processed = 0

    # iterate over all the batches
    for batch_names, batch_files in zip(
        [fovs[i:i + batch_size] for i in range(0, cohort_len, batch_size)],
        [filenames[i:i + batch_size] for i in range(0, cohort_len, batch_size)]
    ):
        # extract the image data for each batch
        start_time = timeit.default_timer()
        if is_mibitiff:
            image_data = load_utils.load_imgs_from_mibitiff(data_dir=tiff_dir,
                                                            mibitiff_files=batch_files,
                                                            dtype=dtype)
        else:
            image_data = load_utils.load_imgs_from_tree(data_dir=tiff_dir,
                                                        img_sub_folder=img_sub_folder,
                                                        fovs=batch_names,
                                                        dtype=dtype)

        end_time = timeit.default_timer()
        print("Time to load image data: %.5f" % (end_time - start_time))

        # get the channel names directly from image_data if not specified
        if chans is None:
            chans = image_data.channels.values

        # verify the channel names (important if the user explicitly specifies channels)
        misc_utils.verify_in_list(
            provided_chans=chans,
            image_chans=image_data.channels.values
        )

        # subset image_data on just the channel names provided
        image_data = image_data.loc[..., chans]

        # run Gaussian blurring per channel
        start_time = timeit.default_timer()
        if gaussian_blur:
            for fov in batch_names:
                for chan in chans:
                    # NOTE: running opencv GaussianBlur with sigmaX=1, borderType=BORDER_REPLICATE,
                    # and ksize=(5, 5) might work even better and may be faster

                    # mode 'nearest' extends input by replicating last pixel
                    # same method that MATLAB imgaussfilter uses
                    # truncate the filter at 2 standard deviations like MATLAB
                    # image_data.loc[fov, :, :, chan] = ndimage.gaussian_filter(
                    #     image_data.loc[fov, :, :, chan], sigma=blur_factor,
                    #     mode='nearest', truncate=2.0
                    # )
                    image_data.loc[fov, :, :, chan] = cv2.GaussianBlur(
                        image_data.loc[fov, :, :, chan].values, sigmaX=1,
                        borderType=cv2.BORDER_REPLICATE, ksize=(5, 5)
                    )
        end_time = timeit.default_timer()
        print("Time to add Gaussian blur: %.5f" % (end_time - start_time))

        # compute the QC metrics for the batch
        start_time = timeit.default_timer()
        df_nonzero_mean_batch = pd.DataFrame(
            compute_nonzero_mean_intensity(image_data), columns=chans
        )
        end_time = timeit.default_timer()
        print("Time to compute non-zero mean intensity: %.5f" % (end_time - start_time))

        start_time = timeit.default_timer()
        df_total_intensity_batch = pd.DataFrame(
            compute_total_intensity(image_data), columns=chans
        )
        end_time = timeit.default_timer()
        print("Time to compute total intensity: %.5f" % (end_time - start_time))

        start_time = timeit.default_timer()
        df_99_9_intensity_batch = pd.DataFrame(
            compute_99_9_intensity(image_data), columns=chans
        )
        end_time = timeit.default_timer()
        print("Time to compute 99.9%% intensity value: %.5f" % (end_time - start_time))

        # append the batch_names as fovs to each DataFrame
        df_nonzero_mean_batch['fov'] = batch_names
        df_total_intensity_batch['fov'] = batch_names
        df_99_9_intensity_batch['fov'] = batch_names

        # append the batch QC metric data to the full processed data
        start_time = timeit.default_timer()
        df_nonzero_mean = pd.concat([df_nonzero_mean, df_nonzero_mean_batch])
        df_total_intensity = pd.concat([df_total_intensity, df_total_intensity_batch])
        df_99_9_intensity = pd.concat([df_99_9_intensity, df_99_9_intensity_batch])
        end_time = timeit.default_timer()
        print("Time to concatenate DataFrame: %.5f" % (end_time - start_time))

        # update number of fovs processed
        fovs_processed += batch_size

        # print fovs processed update
        print("Number of fovs processed: %d" % fovs_processed)

    # reset the indices to make indexing consistent
    df_nonzero_mean = df_nonzero_mean.reset_index(drop=True)
    df_total_intensity = df_total_intensity.reset_index(drop=True)
    df_99_9_intensity = df_99_9_intensity.reset_index(drop=True)

    # create a dictionary mapping the metric name to its respective DataFrame
    qc_data = {
        'nonzero_mean': df_nonzero_mean,
        'total_intensity': df_total_intensity,
        '99_9_intensity': df_99_9_intensity
    }

    return qc_data
