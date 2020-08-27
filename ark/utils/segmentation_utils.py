import os
import copy
import warnings

import scipy.ndimage as nd
import numpy as np
import pandas as pd

import xarray as xr

from skimage.feature import peak_local_max
from skimage.measure import label, regionprops, regionprops_table

import skimage.morphology as morph
from skimage.segmentation import relabel_sequential
import skimage.io as io

from ark.utils import plot_utils, io_utils


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


def visualize_watershed_transform(segmentation_labels_xr, channel_data_xr,
                                  output_dir, model_output,
                                  overlay_channels, fovs=None,
                                  interior_model="pixelwise_interior", interior_threshold=0.25,
                                  interior_smooth=3, maxima_model="pixelwise_interior",
                                  maxima_smooth=3,
                                  maxima_threshold=0.05, nuclear_expansion=None,
                                  randomize_cell_labels=True,
                                  save_tifs='overlays'):
    """Runs the watershed transform over a set of probability masks output by deepcell network
    Inputs:
        segmentation_labels_xr (xarray): xarray containing segmentation labels
        channel_data_xr (xarray): xarray containing TIFs
        output_dir (str): path to directory where the output will be saved
        model_output (xarray): xarray containing the different branch outputs from deepcell
        overlay_channels (Tuple): channels to overlay segmentation output over
        fovs (): field of view
        interior_model (str): Name of model to use to identify maxs in the image
        interior_threshold (float): threshold to cut off interior predictions
        interior_smooth (int): value to smooth the interior predictions
        maxima_model (str): Name of the model to use to predict maxes in the image
        maxima_smooth (int): value to smooth the maxima predictions
        nuclear_expansion (int): optional pixel value by which to expand cells if
            doing nuclear segmentation
        save_tifs (str): flag to control what level of output to save. Must be one of:
            all - saves all tifs
            overlays - saves color overlays and segmentation masks
            none - does not save any tifs
    Outputs:
        Saves xarray to output directory"""

    # error check model selected for local maxima finding in the image

    # loop through all fovs and segment
    if fovs is None:
        fovs = model_output.fovs
    for fov in fovs:
        print("analyzing fov {}".format(fov))

        # generate maxima predictions
        maxima_smoothed = nd.gaussian_filter(model_output.loc[fov, :, :, maxima_model],
                                             maxima_smooth)
        maxima_thresholded = maxima_smoothed
        maxima_thresholded[maxima_thresholded < maxima_threshold] = 0
        maxs = peak_local_max(maxima_thresholded, indices=False, min_distance=5,
                              exclude_border=False)

        # generate interior predictions
        interior_smoothed = nd.gaussian_filter(model_output.loc[fov, :, :, interior_model].values,
                                               interior_smooth)
        interior_mask = interior_smoothed > interior_threshold

        # determine if background is based on network output or an expansion
        if nuclear_expansion is not None:
            interior_mask = morph.dilation(interior_mask,
                                           selem=morph.square(nuclear_expansion * 2 + 1))

        # use maxs to generate seeds for watershed
        markers = label(maxs, connectivity=1)

        # watershed over negative interior mask
        labels = np.array(morph.watershed(-interior_smoothed, markers,
                                          mask=interior_mask, watershed_line=0))

        labels, _, _ = relabel_sequential(labels)

        if randomize_cell_labels:
            random_map = plot_utils.randomize_labels(labels)
        else:
            random_map = labels

        # ignore low-contrast image warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if save_tifs != 'none':
                # save segmentation label map
                for chan_list in overlay_channels:
                    if len(chan_list) == 1:
                        # if only one entry in list, make single channel overlay
                        channel = chan_list[0]
                        chan_marker = channel_data_xr.loc[fov, :, :, channel].values
                        plot_utils.plot_overlay(random_map, plotting_tif=chan_marker,
                                                path=os.path.join(output_dir,
                                                                  "{}_{}_overlay.tiff".format(fov,
                                                                                              channel)))

                    elif len(chan_list) == 2:
                        input_data = np.zeros(
                            (channel_data_xr.shape[1], channel_data_xr.shape[2], 3))
                        input_data[:, :, 1] = channel_data_xr.loc[fov, :, :, chan_list[0]].values
                        input_data[:, :, 2] = channel_data_xr.loc[fov, :, :, chan_list[1]].values
                        plot_utils.plot_overlay(
                            random_map, plotting_tif=input_data,
                            path=os.path.join(
                                output_dir,
                                "{}_{}_{}_overlay.tiff".format(fov, chan_list[0], chan_list[1])))
                    elif len(chan_list) == 3:
                        # if three entries, make a 3 color stack, with third channel in first index (red)
                        input_data = np.zeros(
                            (channel_data_xr.shape[1], channel_data_xr.shape[2], 3))
                        input_data[:, :, 1] = channel_data_xr.loc[fov, :, :, chan_list[0]].values
                        input_data[:, :, 2] = channel_data_xr.loc[fov, :, :, chan_list[1]].values
                        input_data[:, :, 0] = channel_data_xr.loc[fov, :, :, chan_list[2]].values
                        plot_utils.plot_overlay(random_map, plotting_tif=input_data,
                                                path=os.path.join(output_dir,
                                                                  "{}_{}_{}_{}_overlay.tiff".
                                                                  format(fov,
                                                                         chan_list[0],
                                                                         chan_list[1],
                                                                         chan_list[2])))

                io.imsave(os.path.join(output_dir, "{}_segmentation_labels.tiff".format(fov)),
                          random_map)

            if save_tifs == 'all':
                # save borders of segmentation map
                chan_marker = channel_data_xr.loc[fov, :, :, channel_data_xr].values
                plot_utils.plot_overlay(random_map, plotting_tif=chan_marker,
                                        path=os.path.join(output_dir,
                                                          "{}_segmentation_borders.tiff".format(
                                                              fov)))

                plot_utils.plot_overlay(random_map, plotting_tif=chan_marker,
                                        path=os.path.join(output_dir,
                                                          "{}_segmentation_labels.tiff".format(
                                                              fov)))

                io.imsave(os.path.join(output_dir, "{}_interior_smoothed.tiff".format(fov)),
                          interior_smoothed.astype("float32"))

    segmentation_labels_xr.loc[fov, :, :, 'whole_cell'] = random_map

    save_name = os.path.join(output_dir, 'segmentation_labels.xr')
    if os.path.exists(save_name):
        print("overwriting previously generated processed output file")
        os.remove(save_name)

    # segmentation_labels_xr.to_netcdf(save_name, format='NETCDF4')
    segmentation_labels_xr.to_netcdf(save_name, format="NETCDF3_64BIT")
