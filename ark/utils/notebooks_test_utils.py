import json
import os

import feather
import numpy as np
import pandas as pd
import skimage.io as io


def create_pixel_som_files(base_dir,
                           pixel_data_dir,
                           pixel_channel_avg_som_cluster,
                           pixel_channel_avg_meta_cluster,
                           fovs,
                           channels,
                           create_seg_dir=True) -> None:
    """
    Mock the creation of files needed for cell clustering visualization:

    * Pixel consensus data
    * Average channel expression per pixel SOM cluster
    * Average channel expression per pixel meta cluster

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        pixel_data_dir (str): The subdirectory (within `base_dir`) which contains the pixel data.
        pixel_channel_avg_som_cluster (str): The subdirectory (within `base_dir`) which contains
            the `pixel_channel_avg_som_cluster` file.
        pixel_channel_avg_meta_cluster (str): The subdirectory (within `base_dir`) which contains
            the `pixel_channel_avg_meta_cluster` file.
        fovs (list): The list of fovs to use.
        channels (list): The list of channels to use.
        create_seg_dir (bool): Whether to include segmentation labels or not.
    """

    # get path to the data
    data_path = os.path.join(base_dir, pixel_data_dir)

    # make sample consensus data for each fov
    for fov in fovs:
        fov_data = pd.DataFrame(
            np.random.rand(2500, len(channels)),
            columns=channels
        )

        fov_data['fov'] = fov
        fov_data['row_index'] = np.repeat(range(50), 50)
        fov_data['column_index'] = np.tile(range(50), 50)

        if create_seg_dir:
            fov_data['segmentation_label'] = range(1, 2501)

        fov_data['pixel_som_cluster'] = np.repeat(range(1, 101), 25)
        fov_data['pixel_meta_cluster'] = np.repeat(range(1, 21), 125)

        feather.write_dataframe(
            fov_data,
            os.path.join(data_path, '%s.feather' % fov),
            compression='uncompressed'
        )

    # define the average channel expression per pixel SOM cluster
    avg_channels_som = np.random.rand(100, len(channels) + 3)
    avg_channels_som_cols = ['pixel_som_cluster'] + channels + ['count', 'pixel_meta_cluster']
    avg_channels_som = pd.DataFrame(
        avg_channels_som,
        columns=avg_channels_som_cols
    )
    avg_channels_som['pixel_som_cluster'] = range(1, 101)
    avg_channels_som['pixel_meta_cluster'] = np.repeat(range(1, 21), 5)
    avg_channels_som.to_csv(
        os.path.join(base_dir, pixel_channel_avg_som_cluster),
        index=False
    )

    # define the average channel expression per pixel meta cluster
    avg_channels_meta = np.random.rand(20, len(channels) + 2)
    avg_channels_meta_cols = ['pixel_meta_cluster'] + channels + ['count']
    avg_channels_meta = pd.DataFrame(
        avg_channels_meta,
        columns=avg_channels_meta_cols
    )
    avg_channels_meta['pixel_meta_cluster'] = range(1, 21)
    avg_channels_meta.to_csv(
        os.path.join(base_dir, pixel_channel_avg_meta_cluster),
        index=False
    )


def create_pixel_remap_files(base_dir,  pixel_meta_cluster_mapping):
    """
    Generates the pixel_remap file for simulating the metaclustering gui.

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        pixel_meta_cluster_mapping (str): The file location to save the remapped metaclusters to.
    """
    # define the remapping file
    remap_data = pd.DataFrame(
        np.random.rand(100, 3),
        columns=['cluster', 'metacluster', 'mc_name']
    )
    remap_data['cluster'] = range(1, 101)
    remap_data['metacluster'] = np.repeat(range(1, 11), 10)
    remap_data['mc_name'] = np.repeat(['meta_' + str(i) for i in range(1, 11)], 10)
    remap_data.to_csv(os.path.join(base_dir, pixel_meta_cluster_mapping), index=False)


def create_cell_som_files(base_dir,
                          fovs,
                          channels,
                          cell_table_path,
                          cell_data,
                          weighted_cell_channel,
                          cell_som_cluster_count_avgs,
                          cell_meta_cluster_count_avgs,
                          cell_som_cluster_channel_avg,
                          cell_meta_cluster_channel_avg,
                          pixel_cluster_col='pixel_meta_cluster_rename',
                          cell_prefix='test',
                          ):
    """Mock the creation of files needed for cell clustering visualization:

    * Cell table
    * Cell consensus data
    * Weighted channel table
    * Average number of pixel clusters per cell SOM and meta cluster
    * Average weighted channel expression per cell SOM and meta cluster

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        fovs (list): The list of fovs to use.
        channels (list): The list of channels to use.
        cell_table_path (str): The path to the cell table.
        cell_data (str): The path to the `cell_data` file.
        weighted_cell_channel (str): The path to the `weighted_cell_channel` file.
        cell_som_cluster_counts_avgs (str): The path to the `cell_som_cluster_count_avgs` file.
        cell_meta_cluster_count_avgs (str): The path to the `cell_meta_cluster_count_avgs` file.
        cell_som_cluster_channel_avg (str): The path to the `cell_som_cluster_channel_avg` file.
        cell_meta_cluster_channel_avg (str): The path to the `cell_meta_cluster_channel_avg` file.
        pixel_cluster_col (str): The name of the pixel cluster column to aggregate on.
        cell_prefix (str): The prefix to place before each cell clustering directory/file.
    """

    # define the cell table, cell consensus data, and weighted channel tables
    cell_table = pd.DataFrame()
    cell_consensus_data = pd.DataFrame()
    weighted_channel_exp = pd.DataFrame()

    for fov in fovs:
        cell_table_fov = np.random.rand(1000, len(channels) + 3)
        cell_table_fov_cols = ['cell_size'] + channels + ['label', 'fov']
        cell_table_fov = pd.DataFrame(
            cell_table_fov,
            columns=cell_table_fov_cols
        )
        cell_table_fov['label'] = range(1, 1001)
        cell_table_fov['fov'] = fov
        cell_table = pd.concat([cell_table, cell_table_fov])

        cell_consensus_fov = np.random.rand(1000, 25)
        cell_consensus_fov_cols = ['cell_size', 'fov'] + \
            ['%s_' % pixel_cluster_col + str(i) for i in range(1, 21)] + \
            ['segmentation_label', 'cell_som_cluster', 'cell_meta_cluster']
        cell_consensus_fov = pd.DataFrame(
            cell_consensus_fov,
            columns=cell_consensus_fov_cols
        )
        cell_consensus_fov['fov'] = fov
        cell_consensus_fov['segmentation_label'] = range(1, 1001)
        cell_consensus_fov['cell_som_cluster'] = np.repeat(range(1, 101), 10)
        cell_consensus_fov['cell_meta_cluster'] = np.repeat(range(1, 21), 50)
        cell_consensus_data = pd.concat([cell_consensus_data, cell_consensus_fov])

        weighted_channel_fov = np.random.rand(1000, len(channels) + 3)
        weighted_channel_fov_cols = channels + ['cell_size', 'fov', 'segmentation_label']
        weighted_channel_fov = pd.DataFrame(
            weighted_channel_fov,
            columns=weighted_channel_fov_cols
        )
        weighted_channel_fov['fov'] = fov
        weighted_channel_fov['segmentation_label'] = range(1, 1001)
        weighted_channel_exp = pd.concat([weighted_channel_exp, weighted_channel_fov])

    cell_table.to_csv(os.path.join(base_dir, cell_table_path), index=False)

    feather.write_dataframe(
        cell_consensus_data,
        os.path.join(base_dir, cell_data),
        compression='uncompressed'
    )
    weighted_channel_exp.to_csv(
        os.path.join(base_dir, weighted_cell_channel),
        index=False
    )

    # define the average pixel count expresssion per cell SOM cluster
    avg_clusters_som = np.random.randint(1, 64, (100, 23))
    avg_clusters_som_cols = ['cell_som_cluster'] + \
        ['%s_' % pixel_cluster_col + str(i) for i in range(1, 21)] + \
        ['count', 'cell_meta_cluster']
    avg_clusters_som = pd.DataFrame(
        avg_clusters_som,
        columns=avg_clusters_som_cols
    )
    avg_clusters_som['cell_som_cluster'] = range(1, 101)
    avg_clusters_som['cell_meta_cluster'] = np.repeat(range(1, 21), 5)
    avg_clusters_som.to_csv(
        os.path.join(base_dir, cell_som_cluster_count_avgs),
        index=False
    )

    # define the average pixel count expresssion per cell meta cluster
    avg_clusters_meta = np.random.randint(1, 64, (20, 22))
    avg_clusters_meta_cols = ['cell_meta_cluster'] + \
        ['%s_' % pixel_cluster_col + str(i) for i in range(1, 21)] + \
        ['count']
    avg_clusters_meta = pd.DataFrame(
        avg_clusters_meta,
        columns=avg_clusters_meta_cols
    )
    avg_clusters_meta['cell_meta_cluster'] = range(1, 21)
    avg_clusters_meta.to_csv(
        os.path.join(base_dir, cell_meta_cluster_count_avgs),
        index=False
    )

    # define the average weighted channel expression per cell SOM cluster
    avg_channels_som = np.random.rand(100, len(channels) + 2)
    avg_channels_som_cols = ['cell_som_cluster'] + channels + ['cell_meta_cluster']
    avg_channels_som = pd.DataFrame(
        avg_clusters_som,
        columns=avg_channels_som_cols
    )
    avg_channels_som['cell_som_cluster'] = range(1, 101)
    avg_channels_som['cell_meta_cluster'] = np.repeat(range(1, 21), 5)
    avg_channels_som.to_csv(
        os.path.join(base_dir, cell_som_cluster_channel_avg),
        index=False
    )

    # define the average weighted channel expression per cell meta cluster
    avg_channels_meta = np.random.rand(20, len(channels) + 2)
    avg_channels_meta_cols = ['cell_meta_cluster'] + channels
    avg_channels_meta = pd.DataFrame(
        avg_clusters_meta,
        columns=avg_channels_meta_cols
    )
    avg_channels_meta['cell_meta_cluster'] = range(1, 21)
    avg_channels_meta.to_csv(
        os.path.join(base_dir, cell_meta_cluster_channel_avg),
        index=False
    )


def create_cell_remap_files(base_dir,  cell_meta_cluster_remap):
    """
    Generates the cell_remap file for simulating the metaclustering gui.

    Args:
        base_dir (str): The base directory containing all inputs / outputs / data.
        cell_meta_cluster_remap (str): The file location to save the remapped metaclusters to.
    """
    # define the remapping file
    remap_data = pd.DataFrame(
        np.random.rand(100, 3),
        columns=['cluster', 'metacluster', 'mc_name']
    )
    remap_data['cluster'] = range(1, 101)
    remap_data['metacluster'] = np.repeat(range(1, 11), 10)
    remap_data['mc_name'] = np.repeat(['meta_' + str(i) for i in range(1, 11)], 10)
    remap_data.to_csv(os.path.join(base_dir, cell_meta_cluster_remap), index=False)


def generate_sample_feature_tifs(fovs, deepcell_output_dir, img_shape=(50, 50)):
    """Generate a sample _feature_0 and _feature_1 tiff file for each fov.

    Done to bypass the bottleneck of create_deepcell_output, for testing purposes we don't care
    about correct segmentation labels.

    Args:
        fovs (list): The list of fovs to generate sample tiff files for
        deepcell_output_dir (str): The path to the output directory
        img_shape (tuple): Dimensions of the tifs to create
    """

    # generate a random image for each fov, set as both whole cell and nuclear
    for fov in fovs:
        rand_img = np.random.randint(0, 16, size=img_shape)
        io.imsave(os.path.join(deepcell_output_dir, fov + "_feature_0.tiff"), rand_img,
                  check_contrast=False)
        io.imsave(os.path.join(deepcell_output_dir, fov + "_feature_1.tiff"), rand_img,
                  check_contrast=False)
