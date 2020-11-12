import os
import copy
import warnings
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects

from ark.utils import plot_utils, io_utils, misc_utils

import ark.settings as settings


def find_nuclear_label_id(nuc_segmentation_labels, cell_coords):
    """Get the ID of the nuclear mask which has the greatest amount of overlap with a given cell

    Args:
        nuc_segmentation_labels (numpy.ndarray):
            predicted nuclear segmentations
        cell_coords (list):
            list of coords specifying pixels that belong to a cell

    Returns:
        int or None:
            Integer ID of the nuclear mask that overlaps most with cell.
            If no matches found, returns None.
    """

    ids, counts = np.unique(nuc_segmentation_labels[tuple(cell_coords.T)], return_counts=True)

    # Return nuclear ID with greatest overlap. If only 0, return None
    if ids[ids != 0].size == 0:
        nuclear_label_id = None
    else:
        nuclear_label_id = ids[ids != 0][np.argmax(counts[ids != 0])]

    return nuclear_label_id


def split_large_nuclei(cell_segmentation_labels, nuc_segmentation_labels, cell_ids, min_size=5):
    """Splits nuclei that are bigger than the corresponding cell into multiple pieces

    Args:
        cell_segmentation_labels (numpy.ndarray):
            predicted cell segmentations
        nuc_segmentation_labels (numpy.ndarray):
            predicted nuclear segmentations
        cell_ids (numpy.ndarray):
            the unique cells in the segmentation mask
        min_size (int):
            number of pixels of nucleus that must be outside of cell in order to be classified a
            new object. Nuclei with fewer than this many extra pixels will not be relabeled

    Returns:
        numpy.ndarray:
            modified nuclear segmentation mask
    """

    nuc_labels_modified = np.copy(nuc_segmentation_labels)
    max_nuc_id = np.max(nuc_segmentation_labels)

    cell_props = pd.DataFrame(regionprops_table(cell_segmentation_labels,
                                                properties=['label', 'coords']))

    for cell in cell_ids:
        coords = cell_props.loc[cell_props['label'] == cell, 'coords'].values[0]

        nuc_id = find_nuclear_label_id(nuc_segmentation_labels=nuc_segmentation_labels,
                                       cell_coords=coords)

        # only proceed if there's a valid nuc_id
        if nuc_id is not None:
            # figure out if nuclear label is completely contained within cell label
            cell_vals = nuc_segmentation_labels[tuple(coords.T)]
            nuc_count = np.sum(cell_vals == nuc_id)

            nuc_mask = nuc_segmentation_labels == nuc_id

            # only proceed if a non-negligible part of the nucleus is outside of the cell
            if np.sum(nuc_mask) - nuc_count > min_size:
                # relabel nuclear counts within the cell
                cell_mask = cell_segmentation_labels == cell
                new_nuc_mask = np.logical_and(cell_mask, nuc_mask)
                max_nuc_id += 1
                nuc_labels_modified[new_nuc_mask] = max_nuc_id

    nuc_labels_modified = remove_small_objects(ar=nuc_labels_modified, min_size=5)

    return nuc_labels_modified


def transform_expression_matrix(cell_table, transform, transform_kwargs=None):
    """Transform an xarray of marker counts with supplied transformation

    Args:
        cell_table (xarray.DataArray):
            xarray containing marker expression values
        transform (str):
            the type of transform to apply. Must be one of ['size_norm', 'arcsinh']
        transform_kwargs (dict):
            optional dictionary with additional settings for the transforms

    Returns:
        xarray.DataArray:
            xarray of counts per marker normalized by cell size
    """
    valid_transforms = ['size_norm', 'arcsinh']
    misc_utils.verify_in_list(transform=transform, valid_transforms=valid_transforms)

    if transform_kwargs is None:
        transform_kwargs = {}

    # generate array to hold transformed data
    cell_table_transformed = copy.deepcopy(cell_table)

    # get start and end indices of channel data
    channel_start = np.where(cell_table.features == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(cell_table.features == settings.POST_CHANNEL_COL)[0][0]

    if transform == 'size_norm':

        # get the size of each cell
        size_index = np.where(cell_table.features == settings.CELL_SIZE)[0][0]
        cell_size = cell_table[:, :, size_index:size_index + 1].values

        # generate cell_size array that is broadcast to have the same shape as the channels
        cell_size_large = np.repeat(cell_size, channel_end - channel_start, axis=2)

        # Only calculate where cell_size > 0
        cell_table_transformed.values[:, :, channel_start:channel_end] = \
            np.divide(cell_table_transformed.values[:, :, channel_start:channel_end],
                      cell_size_large, where=cell_size_large > 0)

    elif transform == 'arcsinh':
        linear_factor = transform_kwargs.get('linear_factor', 100)

        # first linearly scale the data
        cell_table_transformed.values[:, :, channel_start:channel_end] *= linear_factor

        # arcsinh transformation
        cell_table_transformed.values[:, :, channel_start:channel_end] = \
            np.arcsinh(cell_table_transformed[:, :, channel_start:channel_end].values)

    return cell_table_transformed


def concatenate_csv(base_dir, csv_files, column_name="fov", column_values=None):
    """Take a list of CSV paths and concatenates them together,
    adding in the identifier in column_values

    Saves combined CSV file into the same folder

    Args:
        base_dir (str):
            directory to read and write csv_files into
        csv_files (list):
            a list csv files
        column_name (str):
            optional column name, defaults to fov
        column_values (list):
            optional values to use for each CSV, defaults to csv name
    """

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


def visualize_segmentation(segmentation_labels_xr, channel_data_xr,
                           output_dir, chan_list=None, fovs=None, show=False):
    """For each fov, generates segmentation labels, segmentation borders, and overlays
    over the channels in chan_list if chan_list is provided.

    Saves overlay images to output directory

    Args:
        segmentation_labels_xr (xarray.DataArray):
            xarray containing segmentation labels
        channel_data_xr (xarray.DataArray):
            xarray containing TIFs
        output_dir (str):
            path to directory where the output will be saved
        chan_list (list):
            list of channels to overlay segmentation output over
        fovs (list):
            list of FOVs to subset in segmentation_labels_xr
        show (bool):
            whether or not to show plot
    """

    if fovs is None:
        fovs = segmentation_labels_xr.fovs.values
    misc_utils.verify_in_list(fovs=fovs, segmentation_label_fovs=segmentation_labels_xr.fovs)

    if chan_list is not None:
        misc_utils.verify_in_list(channels=chan_list,
                                  channel_data_channels=channel_data_xr.channels)

    for fov in fovs:
        labels = segmentation_labels_xr.loc[fov, :, :, 'whole_cell'].values
        # If chan_list is provided, overlay segmentation output over it
        if chan_list is not None:
            input_data = channel_data_xr.loc[fov, :, :, chan_list].values
            save_path = '_'.join([f'{fov}', *chan_list.astype('str'), 'overlay.tiff'])
            plot_utils.plot_overlay(
                labels,
                plotting_tif=input_data,
                path=os.path.join(output_dir, save_path), show=show
            )
        # Generates segmentation borders and labels
        plot_utils.plot_overlay(
            labels,
            plotting_tif=None,
            path=os.path.join(output_dir, f'{fov}_segmentation_borders.tiff')
        )
        io.imsave(os.path.join(output_dir, f'{fov}_segmentation_labels.tiff'), labels)
