import multiprocessing
import os
from functools import partial
from shutil import rmtree, move

import feather
from pyarrow.lib import ArrowInvalid
from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers
from ark.phenotyping import pixel_cluster_utils

multiprocessing.set_start_method('spawn', force=True)


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
    move(data_path + '_temp', data_path)


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
