import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns

import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils


def compute_nonzero_mean_intensity(image_data):
    """Compute the nonzero mean of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The nonzero mean intensity of the fov/chan pair (np.nan if the channel contains all 0s)
    """

    # take just the non-zero pixels
    image_data_nonzero = image_data[image_data != 0]

    # take the mean of the non-zero pixels and assign to (fov, channel) in array
    # unless there are no non-zero pixels, in which case default to 0
    if len(image_data_nonzero) > 0:
        nonzero_mean_intensity = image_data_nonzero.mean()
    else:
        nonzero_mean_intensity = 0

    return nonzero_mean_intensity


def compute_total_intensity(image_data):
    """Compute the sum of all pixels of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The total intensity of the fov/chan pair (np.nan if the channel contains all 0s)
    """

    return np.sum(image_data)


def compute_99_9_intensity(image_data):
    """Compute the 99.9% pixel intensity value of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The 99.9% pixel intensity value of a specific fov/chan pair
    """

    return np.percentile(image_data, q=99.9)


def compute_qc_metrics_batch(image_data, fovs, chans, gaussian_blur=False, blur_factor=1):
    """Compute the QC metric matrices for a fov batch

    Helper function to compute_qc_metrics

    Args:
        image_data (xarray.DataArray):
            the data associated with the fov batch
        fovs (list):
            the list of fov names in the batch
        chans (list):
            the subset of channels specified
        gaussian_blur (bool):
            whether to add a Gaussian blur to each batch
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
            ignored if gaussian_blur set to False


    Returns:
        dict:
            A mapping between each QC metric name and their respective DataFrames (batch)
    """

    # subset image_data on just the channel names provided
    image_data = image_data.loc[..., chans]

    # define a numpy array for all the metrics to extract
    # NOTE: numpy array is faster for indexing than pandas
    blank_arr = np.zeros((image_data.shape[0], image_data.shape[3]), dtype='float32')
    nonzero_mean_intensity = copy.deepcopy(blank_arr)
    total_intensity = copy.deepcopy(blank_arr)
    intensity_99_9 = copy.deepcopy(blank_arr)

    # NOTE: looping through each fov and channel separately much faster
    # than numpy vectorization
    for i in np.arange(image_data.shape[0]):
        for j in np.arange(image_data.shape[3]):
            # extract the data for the fov and channel as float
            image_data_np = image_data[i, :, :, j].values.astype('float32')

            # STEP 1: gaussian blur (if specified)
            if gaussian_blur:
                image_data_np = gaussian_filter(
                    image_data_np, sigma=blur_factor, mode='nearest', truncate=2.0
                )

            # STEP 2: extract non-zero mean intensity
            nonzero_mean_intensity[i, j] = compute_nonzero_mean_intensity(image_data_np)

            # STEP 3: extract total intensity
            total_intensity[i, j] = compute_total_intensity(image_data_np)

            # STEP 4: take 99.9% value of the data and assign
            intensity_99_9[i, j] = compute_99_9_intensity(image_data_np)

    # convert the numpy arrays to pandas DataFrames
    df_nonzero_mean_batch = pd.DataFrame(
        nonzero_mean_intensity, columns=chans
    )

    df_total_intensity_batch = pd.DataFrame(
        total_intensity, columns=chans
    )

    df_99_9_intensity_batch = pd.DataFrame(
        intensity_99_9, columns=chans
    )

    # append the batch_names as fovs to each DataFrame
    df_nonzero_mean_batch['fov'] = fovs
    df_total_intensity_batch['fov'] = fovs
    df_99_9_intensity_batch['fov'] = fovs

    # create a dictionary mapping the metric name to its respective DataFrame
    qc_data_batch = {
        'nonzero_mean_batch': df_nonzero_mean_batch,
        'total_intensity_batch': df_total_intensity_batch,
        '99_9_intensity_batch': df_99_9_intensity_batch
    }

    return qc_data_batch


def compute_qc_metrics(tiff_dir, img_sub_folder="TIFs", is_mibitiff=False,
                       fovs=None, chans=None, batch_size=5, gaussian_blur=False,
                       blur_factor=1, dtype='int16'):
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

    # define number of fovs processed (for printing out updates to user)
    fovs_processed = 0

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

        # compute the QC metrics of this batch
        qc_data_batch = compute_qc_metrics_batch(
            image_data, batch_names, chans, gaussian_blur, blur_factor
        )

        # append the batch QC metric data to the full processed data
        df_nonzero_mean = pd.concat(
            [df_nonzero_mean, qc_data_batch['nonzero_mean_batch']]
        )
        df_total_intensity = pd.concat(
            [df_total_intensity, qc_data_batch['total_intensity_batch']]
        )
        df_99_9_intensity = pd.concat(
            [df_99_9_intensity, qc_data_batch['99_9_intensity_batch']]
        )

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


def visualize_qc_metrics(qc_metric_df, metric_name, axes_size=16, wrap=6, dpi=None, save_dir=None):
    """Visualize a barplot of a specific QC metric

    Args:
        qc_metric_df (pandas.DataFrame):
            A QC metric matrix as returned by compute_qc_metrics
        metric_name (str):
            The name of the QC metric, used as the y-axis label
        title_size (int):
            The font size of the title
        axes_size (int):
            The font size of the axes labels
        wrap (int):
            How many plots to display per row
        dpi (int):
            If saving, the resolution of the image to use
            Ignored if save_dir is None
        save_dir (str):
            If saving, the name of the directory to save visualization to
    """

    # catplot allows for easy facets on a barplot
    g = sns.catplot(
        x='fov',
        y=metric_name,
        col='channel',
        col_wrap=wrap,
        data=qc_metric_df,
        kind='bar',
        color='black',
        sharex=True,
        sharey=False
    )

    # per Erin's visualization, don't show the hundreds of fov labels on the x-axis
    _ = plt.xticks([])

    # remove the 'channel =' in each subplot title
    _ = g.set_titles(template='{col_name}')

    # per Erin's visualization remove the default axis title on the y-axis
    # and instead show 'fov' along x-axis and the metric name along the y-axis (overarching)
    _ = g.set_axis_labels('', '')
    _ = g.fig.text(
        x=0.5,
        y=0,
        horizontalalignment='center',
        s='fov',
        size=axes_size
    )
    _ = g.fig.text(
        x=0,
        y=0.5,
        verticalalignment='center',
        s='Non-zero Mean Intensity',
        size=axes_size,
        rotation=90
    )

    # save the figure if specified
    if save_dir is not None:
        misc_utils.save_figure(save_dir, '%s_barplot_stats.png' % metric_name, dpi=dpi)
