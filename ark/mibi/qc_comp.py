import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import xarray as xr

import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils


def compute_nonzero_mean_intensity(image_data):
    """Compute the nonzero mean of each channel per fov

    Args:
        image_data (xarray.DataArray):
            the image data with fov and channel dimensions

    Returns:
        numpy.ndarray:
            Matrix indicating the nonzero mean intensity of each channel per fov
    """

    # mask the data to remove 0 values
    masked_image_data = np.ma.masked_equal(image_data.values, 0)

    # compute the nonzero mean across each fov and channel, convert back to numpy array
    # TODO: if a channel is all 0 this will blow up, assumption that this won't happen for now
    return masked_image_data.mean(axis=(1, 2)).filled()


def compute_total_intensity(image_data):
    """Compute the total intensity of each channel per fov

    Args:
        image_data (xarray.DataArray):
            the image data with fov and channel dimensions

    Returns:
        pandas.DataFrame:
            Matrix indicating the nonzero mean intensity of each channel per fov
    """

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

    # compute the 99.9% intensity value across each fov and channel
    return np.percentile(image_data.values, q=99.9, axis=(1, 2))


def compute_qc_metrics(tiff_dir, img_sub_folder="TIFs", is_mibitiff=False,
                       fovs=None, chans=None, batch_size=5, blur_factor=0, dtype="int16"):
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
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
        dtype (str/type):
            data type of base images

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame:

        - Matrix indicating the nonzero mean intensity of each channel per fov
        - Matrix indicating the total intensity of each channel per fov
        - Matrix indicating the 99.9% intensity value of each channel per fov
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

    # iterate over all the batches
    for batch_names, batch_files in zip(
        [fovs[i:i + batch_size] for i in range(0, cohort_len, batch_size)],
        [filenames[i:i + batch_size] for i in range(0, cohort_len, batch_size)]
    ):
        # extract the image data for each batch
        if is_mibitiff:
            image_data = load_utils.load_imgs_from_mibitiff(data_dir=tiff_dir,
                                                            mibitiff_files=batch_files,
                                                            dtype=dtype)
        else:
            image_data = load_utils.load_imgs_from_tree(data_dir=tiff_dir,
                                                        img_sub_folder=img_sub_folder,
                                                        fovs=batch_names,
                                                        dtype=dtype)

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
        # TODO: check with Erin to see if this is the correct way to Gaussian blur
        for chan in chans:
            image_data.loc[..., chan].values = ndimage.gaussian_filter(
                image_data.loc[..., chan].values, sigma=blur_factor
            )

        # compute nonzero mean across each fov and channel of the batch
        df_nonzero_mean = df_nonzero_mean.append(
            pd.DataFrame(compute_nonzero_mean_intensity(image_data), columns=chans)
        )

        # compute total intensity across each fov and channel of the batch
        df_total_intensity = df_total_intensity.append(
            pd.DataFrame(compute_total_intensity(image_data), columns=chans)
        )

        # compute the 99.9% intensity value across each fov and channel of the batch
        df_99_9_intensity = df_99_9_intensity.append(
            pd.DataFrame(compute_99_9_intensity(image_data), columns=chans)
        )

        # append the batch_names as fovs to each DataFrame
        df_nonzero_mean['fovs'] = batch_names
        df_total_intensity['fovs'] = batch_names
        df_99_9_intensity['fovs'] = batch_names

    return df_nonzero_mean, df_total_intensity, df_99_9_intensity
