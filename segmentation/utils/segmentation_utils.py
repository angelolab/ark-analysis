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


from segmentation.utils import plot_utils, signal_extraction_utils


def watershed_transform(model_output, channel_xr, overlay_channels, output_dir, fovs=None,
                        interior_model="pixelwise_interior", interior_threshold=0.25,
                        interior_smooth=3, maxima_model="pixelwise_interior", maxima_smooth=3,
                        maxima_threshold=0.05, nuclear_expansion=None, randomize_cell_labels=True,
                        save_tifs='overlays'):
    """Runs the watershed transform over a set of probability masks output by deepcell network
    Inputs:
        model_output: xarray containing the different branch outputs from deepcell
        channel_xr: xarray containing TIFs
        output_dir: path to directory where the output will be saved
        interior_model: Name of model to use to identify maxs in the image
        interior_threshold: threshold to cut off interior predictions
        interior_smooth: value to smooth the interior predictions
        maxima_model: Name of the model to use to predict maxes in the image
        maxima_smooth: value to smooth the maxima predictions
        maxima_threshold: threshold to cut off maxima predictions
        nuclear_expansion: optional pixel value by which to expand cells if
            doing nuclear segmentation
        randomize_labels: if true, will randomize the order of the labels put out
            by watershed for easier visualization
        save_tifs: flag to control what level of output to save. Must be one of:
            all - saves all tifs
            overlays - saves color overlays and segmentation masks
            none - does not save any tifs
    Outputs:
        Saves xarray to output directory"""

    # error checking
    if fovs is None:
        fovs = model_output.fovs
    else:
        if np.any(~np.isin(fovs, model_output.coords['fovs'])):
            raise ValueError("Invalid FOVs supplied, not all were found in the model output")

    if len(fovs) == 1:
        # don't subset, will change dimensions
        pass
    else:
        model_output = model_output.loc[fovs, :, :, :]

    if np.any(~np.isin(model_output.fovs.values, channel_xr.fovs.values)):
        raise ValueError("Not all of the FOVs in the model output were found in the channel data")

    # flatten overlay list of lists into single list
    flat_channels = [item for sublist in overlay_channels for item in sublist]
    overlay_in_xr = np.isin(flat_channels, channel_xr.channels)
    if len(overlay_in_xr) != np.sum(overlay_in_xr):
        bad_chan = flat_channels[np.where(~overlay_in_xr)[0][0]]
        raise ValueError("{} was listed as an overlay channel, "
                         "but it is not in the channel data".format(bad_chan))

    if not os.path.isdir(output_dir):
        raise ValueError("output directory does not exist")

    segmentation_labels_xr = \
        xr.DataArray(np.zeros((model_output.shape[:-1] + (1,)), dtype="int16"),
                     coords=[model_output.fovs, range(model_output.shape[1]),
                             range(model_output.shape[2]),
                             ['segmentation_label']],
                     dims=['fovs', 'rows', 'cols', 'channels'])

    # error check model selected for local maxima finding in the image
    model_list = ["pixelwise_interior", "watershed_inner", "watershed_outer",
                  "watershed_argmax", "fgbg_foreground", "pixelwise_sum"]

    if maxima_model not in model_list:
        raise ValueError("Invalid maxima model supplied: {}, "
                         "must be one of {}".format(maxima_model, model_list))
    if maxima_model not in model_output.models:
        raise ValueError("The selected maxima model was not found "
                         "in the model output: {}".format(maxima_model))

    # error check model selected for background delineation in the image
    if interior_model not in model_list:
        raise ValueError("Invalid interior model supplied: {}, "
                         "must be one of {}".format(interior_model, model_list))
    if interior_model not in model_output.models:
        raise ValueError("The selected interior model was not found "
                         "in the model output: {} ".format(interior_model))

    if save_tifs not in ['all', 'overlays', 'none']:
        raise ValueError('Invalid save_tif options, but be either "all", "overlays", or "none"')

    # loop through all fovs and segment
    for fov in model_output.fovs.values:
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

        segmentation_labels_xr.loc[fov, :, :, 'segmentation_label'] = random_map

    save_name = os.path.join(output_dir, 'segmentation_labels.xr')
    if os.path.exists(save_name):
        print("overwriting previously generated processed output file")
        os.remove(save_name)

    # segmentation_labels_xr.to_netcdf(save_name, format='NETCDF4')
    segmentation_labels_xr.to_netcdf(save_name, format="NETCDF3_64BIT")


def combine_segmentation_masks(big_mask_xr, small_mask_xr, size_threshold,
                               output_dir, input_xr, overlay_channels):
    """Takes two xarrays of masks, generated using different parameters,
    and combines them together to produce a single unified mask

    Inputs
        big_mask_xr: xarray optimized for large cells
        small_mask_xr: xarray optimized for small cells
        size_threshold: pixel cutoff for large mask, under which cells will be removed
        output_dir: name of directory to save results
        input_xr: xarray containing channels for overlay
        overlay_channels: channels to overlay segmentation output over"""

    # loop through all masks in large xarray
    for point in range(big_mask_xr.shape[0]):
        labels = big_mask_xr[point, :, :, 0].values

        # for each cell, determine if small than threshold
        for cell in np.unique(labels):
            if np.sum(labels == cell) < size_threshold:
                labels[labels == cell] = 0
        big_mask_xr[point, :, :, 0] = labels

    # loop through all masks in small xarray
    for point in range(small_mask_xr.shape[0]):
        labels = small_mask_xr[point, :, :, 0].values
        big_labels = big_mask_xr[point, :, :, 0].values

        cell_id_offset = np.max(big_labels)
        # loop through all cells in small mask
        for idx, value in enumerate(np.unique(labels)[1:]):

            # if the cell overlaps only with background in big mask, transfer it over
            if len(np.unique(big_labels[labels == value])) == 1:
                if np.unique(big_labels[labels == value]) == 0:
                    big_labels[labels == value] = cell_id_offset + idx

        big_mask_xr[point, :, :, 0] = big_labels

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # loop through modified combined mask
    for point in big_mask_xr.points.values:

        # save segmentation labels
        io.imsave(os.path.join(output_dir, point + "_labels.tiff"),
                  big_mask_xr.loc[point, :, :, "segmentation_label"])

        # save overlay plots
        for channel in overlay_channels:
            plot_utils.plot_overlay(big_mask_xr.loc[point, :, :, "segmentation_label"].values,
                                    input_xr.loc[point, :, :, channel].values,
                                    path=os.path.join(output_dir,
                                                      "{}_{}_overlay.tiff".format(point, channel)))

    # big_mask_xr.to_netcdf(
        # os.path.join(output_dir, "deepcell_output_pixel_processed_segmentation_labels.xr"),
        # format="NETCDF4")
    big_mask_xr.to_netcdf(
        os.path.join(output_dir, "deepcell_output_pixel_processed_segmentation_labels.xr"),
        format="NETCDF3_64BIT")


def compute_marker_counts(input_images, segmentation_masks, nuclear_counts=False):
    """Extract single cell protein expression data from channel TIFs for a single point

        Args:
            input_images (xarray): rows x columns x channels matrix of imaging data
            segmentation_masks (numpy array): rows x columns x compartment matrix of masks
            nuclear_counts: boolean flag to determine whether nuclear counts are returned

        Returns:
            xr_counts: xarray containing segmented data of cells x markers"""

    if type(input_images) is not xr.DataArray:
        raise ValueError("Incorrect data type for ground_truth, expecting xarray")

    if type(segmentation_masks) is not xr.DataArray:
        raise ValueError("Incorrect data type for masks, expecting xarray")

    if input_images.shape[:-1] != segmentation_masks.shape[:-1]:
        raise ValueError("Image data and segmentation masks have different dimensions")

    unique_cell_num = len(np.unique(segmentation_masks.values).astype('int'))

    # create np.array to hold subcellular_loc x channel x cell info
    cell_counts = np.zeros((len(segmentation_masks.compartments), unique_cell_num,
                            len(input_images.channels) + 1))

    col_names = np.concatenate((np.array('cell_size'), input_images.channels), axis=None)
    xr_counts = xr.DataArray(copy.copy(cell_counts),
                             coords=[segmentation_masks.compartments,
                                     np.unique(segmentation_masks.values).astype('int'), col_names],
                             dims=['compartments', 'cell_id', 'features'])

    # get regionprops for each cell
    cell_props = regionprops(segmentation_masks.loc[:, :, 'whole_cell'].values)

    # loop through each cell in mask
    for cell in cell_props:
        # get coords corresponding to current cell. This is much faster than mask-based indexing
        cell_coords = cell.coords.T

        # calculate the total signal intensity within cell
        cell_counts = signal_extraction_utils.default_extraction(cell_coords, input_images)
        xr_counts.loc['whole_cell', cell.label, xr_counts.features[1]:] = cell_counts
        xr_counts.loc['whole_cell', cell.label, xr_counts.features[0]] = cell.area

        if nuclear_counts:
            # only include cell_coords that overlap with a nuclear label
            nuc_coords = []
            nuc_mask = segmentation_masks.loc[:, :, 'nuclear']
            for idx in range(cell_coords.shape[1]):
                if nuc_mask[cell_coords[0, idx], cell_coords[1, idx]] > 0:
                    nuc_coords.append(cell_coords[:, idx])

            # extract data from nuclear coords
            nuc_coords = np.stack(nuc_coords, axis=-1)
            nuc_counts = signal_extraction_utils.default_extraction(nuc_coords, input_images)
            xr_counts.loc['nuclear', cell.label, xr_counts.features[1]:] = nuc_counts
            xr_counts.loc['nuclear', cell.label, xr_counts.features[0]] = nuc_coords.shape[1]

    return xr_counts


def generate_expression_matrix(segmentation_labels, image_data, nuclear_counts=False):
    """Create a matrix of cells by channels with the total counts of each marker in each cell.

    Args:
        segmentation_labels: xarray of shape [fovs, rows, cols, compartment] containing
            segmentation masks for each FOV, potentially across multiple cell compartments
        image_data: xarray containing all of the channel data across all FOVs
        nuclear_counts: boolean flag to determine whether nuclear counts are returned

    Returns:
        pd.DataFrame: marker counts per cell normalized by cell size
        pd.DataFrame: marker counts per cell normalized by cell size and arcsinh transformed
    """

    # initialize data frames
    normalized_data = pd.DataFrame()
    transformed_data = pd.DataFrame()

    if nuclear_counts:
        if 'nuclear' not in segmentation_labels.compartments:
            raise ValueError("Nuclear counts set to True, but not nuclear mask provided")

    # loop over each FOV in the dataset
    for fov in segmentation_labels.fovs.values:
        print("extracting data from {}".format(fov))

        # current mask
        segmentation_label = segmentation_labels.loc[fov, :, :, :]

        # extract the counts per cell for each marker
        cell_data = compute_marker_counts(image_data.loc[fov, :, :, :], segmentation_label,
                                          nuclear_counts=nuclear_counts)

        # remove the cell corresponding to background
        cell_data = cell_data[:, 1:, :]

        # get morphology information
        # TODO: generate nuclear morphology information in addition to cell
        cell_props = regionprops_table(segmentation_label[:, :, 0].values.astype('int16'),
                                       properties=["label", "area", "eccentricity",
                                                   "major_axis_length",
                                                   "minor_axis_length", "perimeter"])
        cell_props = pd.DataFrame(cell_props)

        # create version of data normalized by cell size
        cell_data_norm = copy.deepcopy(cell_data)
        cell_size = cell_data.values[:, :, 0:1]

        # generate cell_size array that is broadcast to have the same shape as the data
        cell_size_large = np.repeat(cell_size, cell_data.shape[2] - 1, axis=2)

        # exclude first column (cell size) from area normalization.
        # Only calculate where cell_size > 0
        cell_data_norm.values[:, :, 1:] = np.divide(cell_data_norm.values[:, :, 1:],
                                                    cell_size_large, where=cell_size_large > 0)

        cell_data_norm_linscale = copy.deepcopy(cell_data_norm)
        cell_data_norm_linscale.values[:, :, 1:] = cell_data_norm_linscale.values[:, :, 1:] * 100

        # arcsinh transformation
        cell_data_norm_trans = copy.deepcopy(cell_data_norm_linscale)
        cell_data_norm_trans.values[:, :, 1:] = np.arcsinh(cell_data_norm_trans[:, :, 1:])

        transformed = pd.DataFrame(data=cell_data_norm_trans.values[0, :, :],
                                   columns=cell_data.features)
        transformed = pd.concat([transformed, cell_props], axis=1)
        transformed['fov'] = fov
        transformed_data = transformed_data.append(transformed)

        normalized = pd.DataFrame(data=cell_data_norm.values[0, :, :], columns=cell_data.features)
        normalized = pd.concat([normalized, cell_props], axis=1)
        normalized['fov'] = fov
        normalized_data = normalized_data.append(normalized)

    return normalized_data, transformed_data


def concatenate_csv(base_dir, csv_files, column_name="point", column_values=None):
    """Take a list of CSV paths and concatenates them together,
    adding in the identifier in column_values

    Inputs:
        base_dir: directory to read and write csv_files into
        csv_files: a list csv files
        column_name: optional column name, defaults to point
        column_values: optional values to use for each CSV, defaults to csv name

    Outputs: saved combined csv into same folder"""

    if column_values is None:
        column_values = copy.copy(csv_files)
        column_values = [val.split(".")[0] for val in column_values]

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
