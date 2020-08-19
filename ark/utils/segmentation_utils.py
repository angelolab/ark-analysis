import os
import copy

import numpy as np
import pandas as pd

from ark.utils import io_utils


def find_nuclear_mask_id(nuc_segmentation_mask, cell_coords):
    """Get the ID of the nuclear mask which has the greatest amount of overlap with a given cell

    Args:
        nuc_segmentation_mask (numpy): label mask of nuclear segmentations
        cell_coords (list): list of coords specifying pixels that belong to a cell

    Returns:
        nuclear_mask_id (int): ID of the nuclear mask that overlaps most with cell.
                               If no matches found, returns None.
    """

    ids, counts = np.unique(nuc_segmentation_mask[tuple(cell_coords.T)], return_counts=True)

    # Return nuclear ID with greatest overlap. If only 0, return None
    if ids[ids != 0].size == 0:
        nuclear_mask_id = None
    else:
        nuclear_mask_id = ids[ids != 0][np.argmax(counts[ids != 0])]

    return nuclear_mask_id


def transform_expression_matrix(cell_data, transform, transform_kwargs=None):
    """Transform an xarray of marker counts with supplied transformation

    Args:
        cell_data (xarray): xarray containing marker expression values
        transform (str): the type of transform to apply. Must be one of ['size_norm', 'arcsinh']
        transform_kwargs (dict): optional dictionary with additional settings for the transforms

    Returns:
        cell_data_norm (xarray): counts per marker normalized by cell size
    """
    valid_transforms = ['size_norm', 'arcsinh']

    if transform not in valid_transforms:
        raise ValueError('Invalid transform supplied')

    if transform_kwargs is None:
        transform_kwargs = {}

    # generate array to hold transformed data
    cell_data_transformed = copy.deepcopy(cell_data)

    # get start and end indices of channel data. We skip the 0th entry, which is cell size
    channel_start = 1

    # we include columns up to 'label', which is the first non-channel column
    channel_end = np.where(cell_data.features == 'label')[0][0]

    if transform == 'size_norm':

        # get the size of each cell
        cell_size = cell_data.values[:, :, 0:1]

        # generate cell_size array that is broadcast to have the same shape as the channels
        cell_size_large = np.repeat(cell_size, channel_end - channel_start, axis=2)

        # Only calculate where cell_size > 0
        cell_data_transformed.values[:, :, channel_start:channel_end] = \
            np.divide(cell_data_transformed.values[:, :, channel_start:channel_end],
                      cell_size_large, where=cell_size_large > 0)

    elif transform == 'arcsinh':
        linear_factor = transform_kwargs.get('linear_factor', 100)

        # first linearly scale the data
        cell_data_transformed.values[:, :, channel_start:channel_end] *= linear_factor

        # arcsinh transformation
        cell_data_transformed.values[:, :, channel_start:channel_end] = \
            np.arcsinh(cell_data_transformed[:, :, channel_start:channel_end].values)

    return cell_data_transformed


def concatenate_csv(base_dir, csv_files, column_name="point", column_values=None):
    """Take a list of CSV paths and concatenates them together,
    adding in the identifier in column_values

    Inputs:
        base_dir (str): directory to read and write csv_files into
        csv_files (list): a list csv files
        column_name (str): optional column name, defaults to point
        column_values (list): optional values to use for each CSV, defaults to csv name

    Outputs: saved combined csv into same folder"""

    if column_values is None:
        column_values = io_utils.extract_delimited_names(csv_files, delimiter='.')

    if len(column_values) != len(csv_files):
        raise ValueError("csv_files and column_values have different lengths: "
                         "csv {}, column_values {}".format(len(csv_files), len(column_values)))

    for idx, file in enumerate(csv_files):
        if idx == 0:
            # first one, create master array
            temp_data = pd.read_csv(os.path.join(base_dir, file), header=0, sep=",")
            temp_data[column_name] = column_values[idx]
            combined_data = temp_data
        else:
            temp_data = pd.read_csv(os.path.join(base_dir, file), header=0, sep=",")
            temp_data[column_name] = column_values[idx]
            combined_data = pd.concat((combined_data, temp_data), axis=0, ignore_index=True)

    combined_data.to_csv(os.path.join(base_dir, "combined_data.csv"), index=False)


def visualize_watershed_transform(overaly_channels, channel_xr, random_map,fov,output_dir,save_tifs='overlays'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if save_tifs != 'none':
            # save segmentation label map
            io.imsave(os.path.join(output_dir, "{}_segmentation_labels.tiff".format(fov)),
                      random_map)

        if save_tifs == 'all':
            # save borders of segmentation map
            plot_utils.plot_overlay(random_map, plotting_tif=None,
                                    path=os.path.join(output_dir,
                                                      "{}_segmentation_borders.tiff".format(
                                                          fov)))

            io.imsave(os.path.join(output_dir, "{}_interior_smoothed.tiff".format(fov)),
                      interior_smoothed.astype("float32"))

            io.imsave(os.path.join(output_dir, "{}_maxs_smoothed_thresholded.tiff".format(fov)),
                      maxima_thresholded.astype("float32"))

            io.imsave(os.path.join(output_dir, "{}_maxs.tiff".format(fov)),
                      maxs.astype('uint8'))

            for chan in channel_xr.channels.values:
                io.imsave(os.path.join(output_dir, "{}_{}.tiff".format(fov, chan)),
                          channel_xr.loc[fov, :, :, chan].astype('float32'))

    # plot list of supplied markers overlaid by segmentation mask to assess accuracy
    if save_tifs != 'none':
        for chan_list in overlay_channels:
            if len(chan_list) == 1:
                # if only one entry in list, make single channel overlay
                channel = chan_list[0]
                chan_marker = channel_xr.loc[fov, :, :, channel].values
                plot_utils.plot_overlay(random_map, plotting_tif=chan_marker,
                                        path=os.path.join(output_dir,
                                                          "{}_{}_overlay.tiff".format(fov,
                                                                                      channel)))

            elif len(chan_list) == 2:
                # if two entries, make 2-color stack, skipping 0th index which is red
                input_data = np.zeros((channel_xr.shape[1], channel_xr.shape[2], 3))
                input_data[:, :, 1] = channel_xr.loc[fov, :, :, chan_list[0]].values
                input_data[:, :, 2] = channel_xr.loc[fov, :, :, chan_list[1]].values
                plot_utils.plot_overlay(
                    random_map, plotting_tif=input_data,
                    path=os.path.join(
                        output_dir,
                        "{}_{}_{}_overlay.tiff".format(fov, chan_list[0], chan_list[1])))
            elif len(chan_list) == 3:
                # if three entries, make a 3 color stack, with third channel in first index (red)
                input_data = np.zeros((channel_xr.shape[1], channel_xr.shape[2], 3))
                input_data[:, :, 1] = channel_xr.loc[fov, :, :, chan_list[0]].values
                input_data[:, :, 2] = channel_xr.loc[fov, :, :, chan_list[1]].values
                input_data[:, :, 0] = channel_xr.loc[fov, :, :, chan_list[2]].values
                plot_utils.plot_overlay(random_map, plotting_tif=input_data,
                                        path=os.path.join(output_dir,
                                                          "{}_{}_{}_{}_overlay.tiff".
                                                          format(fov,
                                                                  chan_list[0],
                                                                  chan_list[1],
                                                                  chan_list[2])))

    segmentation_labels_xr.loc[fov, :, :, 'whole_cell'] = random_map

    save_name = os.path.join(output_dir, 'segmentation_labels.xr')
    if os.path.exists(save_name):
      print("overwriting previously generated processed output file")
      os.remove(save_name)
    segmentation_labels_xr.to_netcdf(save_name, format='NETCDF4')
    segmentation_labels_xr.to_netcdf(save_name, format="NETCDF3_64BIT")
