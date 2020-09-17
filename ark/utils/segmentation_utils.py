import os
import copy

import numpy as np
import pandas as pd

from skimage.measure import regionprops_table

from ark.utils import io_utils


def find_nuclear_mask_id(nuc_segmentation_mask, cell_coords):
    """Get the ID of the nuclear mask which has the greatest amount of overlap with a given cell

    Args:
        nuc_segmentation_mask (numpy.ndarray):
            label mask of nuclear segmentations
        cell_coords (list):
            list of coords specifying pixels that belong to a cell

    Returns:
        int or None:
            Integer ID of the nuclear mask that overlaps most with cell.
            If no matches found, returns None.
    """

    ids, counts = np.unique(nuc_segmentation_mask[tuple(cell_coords.T)], return_counts=True)

    # Return nuclear ID with greatest overlap. If only 0, return None
    if ids[ids != 0].size == 0:
        nuclear_mask_id = None
    else:
        nuclear_mask_id = ids[ids != 0][np.argmax(counts[ids != 0])]

    return nuclear_mask_id


def split_large_nuclei(cell_segmentation_mask, nuc_segmentation_mask, cell_ids):
    """Splits nuclei that are bigger than the corresponding cell into multiple pieces

    Args:
        cell_segmentation_mask (numpy.ndarray):
            mask of cell segmentations
        nuc_segmentation_mask (numpy.ndarray):
            mask of nuclear segmentations
        cell_ids (numpy.ndarray):
            the unique cells in the segmentation mask

    Returns:
        numpy.ndarray:
            modified nuclear segmentation mask
    """

    nuc_mask_modified = np.copy(nuc_segmentation_mask)
    max_nuc_id = np.max(nuc_segmentation_mask)

    cell_props = pd.DataFrame(regionprops_table(cell_segmentation_mask,
                                                properties=['label', 'coords']))

    for cell in cell_ids:
        coords = cell_props.loc[cell_props['label'] == cell, 'coords'].values[0]

        nuc_id = find_nuclear_mask_id(nuc_segmentation_mask=nuc_segmentation_mask,
                                      cell_coords=coords)

        # only proceed if there's a valid nuc_id
        if nuc_id is not None:
            # figure out if nuclear label is completely contained within cell label
            cell_vals = nuc_segmentation_mask[tuple(coords.T)]
            nuc_count = np.sum(cell_vals == nuc_id)

            nuc_mask = nuc_segmentation_mask == nuc_id

            # only proceed if parts of the nucleus are outside of the cell
            if nuc_count != np.sum(nuc_mask):
                # relabel nuclear counts within the cell
                cell_mask = cell_segmentation_mask == cell
                new_nuc_mask = np.logical_and(cell_mask, nuc_mask)
                max_nuc_id += 1
                nuc_mask_modified[new_nuc_mask] = max_nuc_id

    return nuc_mask_modified


def transform_expression_matrix(cell_data, transform, transform_kwargs=None):
    """Transform an xarray of marker counts with supplied transformation

    Args:
        cell_data (xarray.DataArray):
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

    Saves combined CSV file into the same folder

    Args:
        base_dir (str):
            directory to read and write csv_files into
        csv_files (list):
            a list csv files
        column_name (str):
            optional column name, defaults to point
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
