import os
import skimage.measure
import copy
import warnings

import numpy as np
import xarray as xr
from skimage.feature import peak_local_max
import skimage.morphology as morph
from skimage.segmentation import relabel_sequential
import skimage.io as io
import pandas as pd
import scipy.ndimage as nd

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

    segmentation_labels_xr = xr.DataArray(np.zeros((model_output.shape[:-1] + (1,)), dtype="int16"),
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
        markers = skimage.measure.label(maxs, connectivity=1)

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


def segment_images(input_images, segmentation_masks):
    """Extract single cell protein expression data from channel TIFs for a single point

        Args:
            input_images (xarray): rows x columns x channels matrix of imaging data
            segmentation_masks (numpy array): rows x columns x mask_type matrix of segmentation data

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
    cell_counts = np.zeros((len(segmentation_masks.subcell_loc), unique_cell_num,
                            len(input_images.channels) + 1))

    col_names = np.concatenate((np.array('cell_size'), input_images.channels), axis=None)
    xr_counts = xr.DataArray(copy.copy(cell_counts),
                             coords=[segmentation_masks.subcell_loc,
                                     np.unique(segmentation_masks.values).astype('int'), col_names],
                             dims=['subcell_loc', 'cell_id', 'features'])

    # loop through each segmentation mask
    for subcell_loc in segmentation_masks.subcell_loc:
        # get regionprops for each mask
        cell_props = skimage.measure.regionprops(segmentation_masks.loc[:, :, subcell_loc].values)

        # loop through each cell in mask
        for cell in cell_props:
            # get coords corresponding to current cell
            cell_coords = cell.coords.T

            # calculate the total signal intensity within cell
            channel_counts = signal_extraction_utils.default_extraction(cell_coords, input_images)
            xr_counts.loc[subcell_loc, cell.label, xr_counts.features[1]:] = channel_counts
            xr_counts.loc[subcell_loc, cell.label, xr_counts.features[0]] = cell.area

    return xr_counts


def extract_single_cell_data(segmentation_labels, image_data,
                             nuc_probs=None):
    """Extract single cell data from a set of images with provided segmentation mask
    Input:
        segmentation_labels: xarray containing a segmentation mask for each point
        image_data: xarray containing the imaging data for each point
        save_dir: path to where the data  will be saved
        nuc_probs: xarray of deepcell_pixel nuclear probabilities for subcellular segmentation
    Output:
        saves output to save_dir"""

    normalized_data = pd.DataFrame()
    transformed_data = pd.DataFrame()

    for fov in segmentation_labels.fovs.values:
        print("extracting data from {}".format(fov))

        segmentation_label = segmentation_labels.loc[fov, :, :, "segmentation_label"]

        # if nuclear probabilities supplied, perform subcellular segmentation
        if nuc_probs is not None:
            # generate nuclear-specific mask for subcellular segmentation
            nuc_prob = nuc_probs.loc[fov, :, :, "nuclear_interior_smoothed"]
            nuc_mask = nuc_prob > 0.3

            # duplicate whole cell data, then subtract nucleus for cytoplasm
            cyto_label = copy.deepcopy(segmentation_label)
            cyto_label[nuc_mask] = 0

            # nuclear data
            nuc_label = copy.deepcopy(segmentation_label)
            nuc_label[~nuc_mask] = 0

            # signal assigned to bg
            bg_label = np.zeros(segmentation_label.shape)
            bg_label[segmentation_label == 0] = 1

            # save different masks to single object
            masks = np.zeros((segmentation_label.shape[0],
                              segmentation_label.shape[1], 4), dtype="int16")
            masks[:, :, 0] = segmentation_label
            masks[:, :, 1] = nuc_label
            masks[:, :, 2] = cyto_label
            masks[:, :, 3] = bg_label

            segmentation_masks = xr.DataArray(copy.copy(masks),
                                              coords=[range(segmentation_labels.shape[1]),
                                                      range(segmentation_labels.shape[2]),
                                                      ['cell_mask', 'nuc_mask',
                                                       'cyto_mask', 'bg_mask']],
                                              dims=['rows', 'cols', 'subcell_loc'])

        # otherwise, just extract a single sum for each unique cell in the image
        else:
            masks = np.expand_dims(segmentation_label.values, axis=-1)
            masks = masks.astype('int16')
            segmentation_masks = xr.DataArray(masks, coords=[range(segmentation_label.shape[0]),
                                                             range(segmentation_label.shape[1]),
                                                             ['cell_mask']],
                                              dims=['rows', 'cols', 'subcell_loc'])

        # segment images based on supplied masks
        cell_data = segment_images(image_data.loc[fov, :, :, :], segmentation_masks)
        cell_data = cell_data[:, 1:, :]

        cell_props = skimage.measure.regionprops_table(
            segmentation_masks[:, :, 0].values.astype('int16'),
            properties=["label", "area", "eccentricity", "major_axis_length",
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
