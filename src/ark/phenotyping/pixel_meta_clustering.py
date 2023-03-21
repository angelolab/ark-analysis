import multiprocessing
import os
import random
from functools import partial
from shutil import rmtree, move

import feather
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers
from ark.phenotyping import pixel_cluster_utils

multiprocessing.set_start_method('spawn', force=True)


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
                            multiprocess=False, batch_size=5, seed=42, overwrite=False):
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
        overwrite (bool):
            If set, force overwrites the meta labels in all the FOVs

    Returns:
        cluster_helpers.PixieConsensusCluster:
            The consensus cluster object containing the SOM to meta mapping
    """

    # define the paths to the data
    pixel_data_path = os.path.join(base_dir, data_dir)
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)

    # path validation
    io_utils.validate_paths([pixel_data_path, som_cluster_avg_path])

    # if overwrite flag set, run on all FOVs in data_dir
    if overwrite:
        print('Overwrite flag set, reassigning meta cluster labels to all FOVs')
        os.mkdir(pixel_data_path + '_temp')
        fovs_list = io_utils.remove_file_extensions(
            io_utils.list_files(pixel_data_path, substrs='.feather')
        )
    # otherwise, only assign meta clusters to FOVs that don't already have them
    else:
        fovs_list = pixel_cluster_utils.find_fovs_missing_col(
            base_dir, data_dir, 'pixel_meta_cluster'
        )

    # make sure fovs_list only contain fovs that exist in the master fovs list specified
    fovs_list = list(set(fovs_list).intersection(fovs))

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
    move(pixel_data_path + '_temp', pixel_data_path)

    return pixel_cc


def generate_meta_avg_files(fovs, channels, base_dir, pixel_cc, data_dir='pixel_mat_data',
                            pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                            pc_chan_avg_meta_cluster_name='pixel_channel_avg_meta_cluster.csv',
                            num_fovs_subset=100, seed=42, overwrite=False):
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
        overwrite (bool):
            If set, force overwrites the existing average channel expression file if it exists
    """

    # define the paths to the data
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)
    meta_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_meta_cluster_name)

    # path validation
    io_utils.validate_paths([som_cluster_avg_path])

    # if the channel meta average file already exists and the overwrite flag isn't set, skip
    if os.path.exists(meta_cluster_avg_path):
        if not overwrite:
            print("Already generated meta cluster channel average file, skipping")
            return

        print("Overwrite flag set, regenerating meta cluster channel average file")

    # compute average channel expression for each pixel meta cluster
    # and the number of pixels per meta cluster
    print("Computing average channel expression across pixel meta clusters")
    pixel_channel_avg_meta_cluster = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
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
        required_cols=['pixel_som_cluster', 'pixel_meta_cluster', 'pixel_meta_cluster_rename']
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
        fov_list = pixel_cluster_utils.find_fovs_missing_col(
            base_dir, pixel_data_dir, 'pixel_meta_cluster_rename'
        )
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
    move(pixel_data_path + '_temp', pixel_data_path)


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
    new_meta_clusters = pd.read_csv(pixel_remapped_path)['pixel_meta_cluster'].unique()

    # read in the remapping
    # TODO: define a separate function for this duplicated logic, OOP will help greatly (soon...)
    pixel_remapped_data = pd.read_csv(pixel_remapped_path)

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
    pixel_channel_avg_meta_cluster = pixel_cluster_utils.compute_pixel_cluster_channel_avg(
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
