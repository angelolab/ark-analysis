import os
import skimage.measure
import copy
import warnings


import numpy as np
import xarray as xr
from skimage.feature import peak_local_max
import skimage.morphology as morph
import skimage.io as io
import pandas as pd

from segmentation.utils import plot_utils


def watershed_transform(pixel_xr, channel_xr, overlay_channels, output_dir, background_threshold=0.25,
                        watershed_xr=None, watershed_maxs=False, watershed_smooth=None, pixel_background=True,
                        pixel_smooth=None,
                        nuclear_expansion=None, randomize_cell_labels=True, small_seed_cutoff=5, rescale_factor=1):

    """Runs the watershed transform over a set of probability masks output by deepcell network
    Inputs:
        pixel_xr: xarray containing the pixel 3-class probabilities
        watershed_xr: xarray containgin the watershed network argmax probabilities
        channel_xr: xarray containing the TIFs of segmentation data
        output_dir: path to directory where the output will be saved
        watershed_maxs: if true, uses the output of watershed network as seeds, otherwise finds local maxs from pixel
        watershed_smooth: Name of smooth mask to use. If unspecified, uses last mask in watershed xr
        pixel_background: if true, will use the interior_mask of the pixel network as the space to watershed over.
            if false, will use the output of the watershed network
        pixel_smooth: Name of smooth mask to use. If unspecicied, uses last mask in pixel xr
        nuclear_expansion: optional pixel value by which to expand cells if doing nuclear segmentation
        randomize_labels: if true, will randomize the order of the labels put out by watershed for easier visualization
        small_seed_cutoff: area threshold for cell seeds, smaller values will be discarded
    Outputs:
        Saves xarray to output directory"""

    # error checking
    if watershed_xr is not None:
        if pixel_xr.shape[0] != watershed_xr.shape[0]:
            raise ValueError("The pixel and watershed xarrays have a different number of points")

        if watershed_smooth is None:
            watershed_smooth = watershed_xr.masks.values[-1]
        else:
            if watershed_smooth not in watershed_xr.masks.values:
                raise ValueError("Invalid watershed smooth name: {} not found in xr".format(watershed_smooth))

    if np.sum(~np.isin(pixel_xr.points.values, channel_xr.points.values)) > 0:
        raise ValueError("Not all of the points in the deepcell output were found in the channel xr")

    overlay_in_xr = np.isin(overlay_channels, channel_xr.channels)
    if len(overlay_in_xr) != np.sum(overlay_in_xr):
        bad_chan = overlay_channels[np.where(~overlay_in_xr)[0][0]]
        raise ValueError("{} was listed as an overlay channel, but it is not in the channel xarray".format(bad_chan))

    if not os.path.isdir(output_dir):
        raise ValueError("output directory does not exist")

    segmentation_labels_xr = xr.DataArray(np.zeros((pixel_xr.shape[:-1] + (1,)), dtype="int16"),
                                          coords=[pixel_xr.points, range(pixel_xr.shape[1]), range(pixel_xr.shape[2]),
                                                  ['segmentation_label']], dims=['points', 'rows', 'cols', 'channels'])

    if pixel_smooth is None:
        pixel_smooth = pixel_xr.masks.values[-1]
    else:
        if pixel_smooth not in pixel_xr.masks.values:
            raise ValueError("Invalid pixel smooth name: {} not found in xr".format(pixel_smooth))

    for point in pixel_xr.points.values:
        print("analyzing point {}".format(point))

        # generate maxs from watershed or pixel
        if watershed_maxs:
            # get local maxima from watershed network
            watershed_smoothed = watershed_xr.loc[point, :, :, watershed_smooth]
            maxs = watershed_smoothed > 2
        else:
            # get local maxima from pixel network
            maxs = peak_local_max(pixel_xr.loc[point, :, :, pixel_smooth].values, indices=False, min_distance=5)

        # generate background mask from watershed or pixel
        if pixel_background:
            # use interior probability from pixel network as space to watershed over
            pixel_smoothed = pixel_xr.loc[point, :, :, pixel_smooth]
            max = np.max(pixel_smoothed.values)
            contour_mask = -pixel_smoothed
            interior_mask = pixel_smoothed > (background_threshold * max)

        else:
            # use watershed network output as space to watershed over
            watershed_smoothed = watershed_xr.loc[point, :, :, watershed_smooth]
            contour_mask = -watershed_smoothed
            interior_mask = watershed_smoothed > 0

        # determine if background is based on network output or an expansion
        if nuclear_expansion is not None:
            interior_mask = morph.dilation(interior_mask, selem=morph.square(nuclear_expansion * 2 + 1))

        # use maxs to generate seeds for watershed
        markers = skimage.measure.label(maxs, connectivity=1)

        # remove any maxs that are 4 pixels or smaller, if maxs are generated from watershed network
        if small_seed_cutoff is not None and watershed_maxs:

            # get size of all seeds
            seed_props = skimage.measure.regionprops(markers, cache=False)
            seed_area = np.array([r.area for r in seed_props])

            # for all seeds less than cutoff, set corresponding pixels to 0
            remove_ids = np.where(seed_area < small_seed_cutoff)

            for id in remove_ids[0]:
                # increment by 1 since regionprops starts counting at 0
                markers[markers == id + 1] = 0

        # watershed over negative interior mask
        labels = np.array(morph.watershed(contour_mask, markers, mask=interior_mask, watershed_line=0))

        if randomize_cell_labels:
            random_map = plot_utils.randomize_labels(labels)
        else:
            random_map = labels

        # ignore low-contrast image warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(output_dir, point + "_segmentation_labels.tiff"), random_map)

        segmentation_labels_xr.loc[point, :, :, 'segmentation_label'] = random_map

        # plot list of supplied markers overlaid by segmentation mask to assess accuracy
        for channel in overlay_channels:
            chan_marker = channel_xr.loc[point, :, :, channel].values
            plot_utils.plot_overlay(random_map, plotting_tif=chan_marker, rescale_factor=rescale_factor,
                                    path=os.path.join(output_dir, point + "_{}_overlay.tiff".format(channel)))

    segmentation_labels_xr.name = pixel_xr.name + "_segmentation_labels"
    segmentation_labels_xr.to_netcdf(os.path.join(output_dir, segmentation_labels_xr.name + ".nc"), format='NETCDF4')


def combine_segmentation_masks(big_mask_xr, small_mask_xr, size_threshold, output_dir, input_xr, overlay_channels):
    """Takes two xarrays of masks, generated using different parameters, and combines them together to produce
        a single unified mask

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
        io.imsave(os.path.join(output_dir, point + "_labels.tiff"), big_mask_xr.loc[point, :, :, "segmentation_label"])

        # save overlay plots
        for channel in overlay_channels:
            plot_utils.plot_overlay(big_mask_xr.loc[point, :, :, "segmentation_label"].values,
                                    input_xr.loc[point, :, :, channel].values,
                                    path=os.path.join(output_dir, "{}_{}_overlay.tiff".format(point, channel)))

    big_mask_xr.to_netcdf(os.path.join(output_dir, "deepcell_output_pixel_processed_segmentation_labels.nc"),
                          format="NETCDF4")


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
    cell_counts = np.zeros((len(segmentation_masks.subcell_loc), unique_cell_num, len(input_images.channels) + 1))

    col_names = np.concatenate((np.array('cell_size'), input_images.channels), axis=None)
    xr_counts = xr.DataArray(copy.copy(cell_counts), coords=[segmentation_masks.subcell_loc,
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

            # calculate the total signal intensity within that cell mask across all channels, and save to numpy
            channel_index = input_images.values[tuple(cell_coords)]
            channel_counts = np.sum(channel_index, axis=0)
            xr_counts.loc[subcell_loc, cell.label, xr_counts.features[1]:] = channel_counts
            xr_counts.loc[subcell_loc, cell.label, xr_counts.features[0]] = cell.area

    return xr_counts


def extract_single_cell_data(segmentation_labels, image_data, save_dir, nuc_probs=None, save_FCS=False):

    """Extract single cell data from a set of images with provided segmentation mask
    Input:
        segmentation_labels: xarray containing a segmentation mask for each point
        image_data: xarray containing the imaging data for each point
        save_dir: path to where the data  will be saved
        nuc_probs: xarray of deepcell_pixel nuclear probabilities for subcellular segmentation
    Output:
        saves output to save_dir"""

    for point in segmentation_labels.points.values:
        print("extracting data from {}".format(point))

        segmentation_label = segmentation_labels.loc[point, :, :, "segmentation_label"]

        # if nuclear probabilities supplied, perform subcellular segmentation
        if nuc_probs is not None:
            # generate nuclear-specific mask for subcellular segmentation
            nuc_prob = nuc_probs.loc[point, :, :, "nuclear_interior_smoothed"]
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
            masks = np.zeros((segmentation_label.shape[0], segmentation_label.shape[1], 4), dtype="int16")
            masks[:, :, 0] = segmentation_label
            masks[:, :, 1] = nuc_label
            masks[:, :, 2] = cyto_label
            masks[:, :, 3] = bg_label

            segmentation_masks = xr.DataArray(copy.copy(masks), coords=[range(segmentation_labels.shape[1]),
                                                                        range(segmentation_labels.shape[2]),
                                                                        ['cell_mask', 'nuc_mask', 'cyto_mask', 'bg_mask']],
                                              dims=['rows', 'cols', 'subcell_loc'])

        # otherwise, just extract a single sum for each unique cell in the image
        else:
            masks = np.expand_dims(segmentation_label.values, axis=-1)
            masks = masks.astype('int16')
            segmentation_masks = xr.DataArray(masks, coords=[range(segmentation_label.shape[0]),
                                                             range(segmentation_label.shape[1]), ['cell_mask']],
                                              dims=['rows', 'cols', 'subcell_loc'])

        # segment images based on supplied masks
        cell_data = segment_images(image_data.loc[point, :, :, :], segmentation_masks)
        cell_data = cell_data[:, 1:, :]

        cell_props = skimage.measure.regionprops_table(segmentation_masks[:, :, 0].values.astype('int16'),
                                                       properties=["label", "area", "eccentricity", "major_axis_length",
                                                                   "minor_axis_length", "perimeter"])
        cell_props = pd.DataFrame(cell_props)


        # create version of data normalized by cell size
        cell_data_norm = copy.deepcopy(cell_data)
        cell_size = cell_data.values[:, :, 0:1]

        # generate cell_size array that is broadcast to have the same shape as the data
        cell_size_large = np.repeat(cell_size, cell_data.shape[2] - 1, axis=2)

        # exclude first column (cell size) from area normalization. Only calculate where cell_size > 0
        cell_data_norm.values[:, :, 1:] = np.divide(cell_data_norm.values[:, :, 1:], cell_size_large, where=cell_size_large > 0)

        cell_data_norm_linscale = copy.deepcopy(cell_data_norm)
        cell_data_norm_linscale.values[:, :, 1:] = cell_data_norm_linscale.values[:, :, 1:] * 100

        # arcsinh transformation
        cell_data_norm_trans = copy.deepcopy(cell_data_norm_linscale)
        cell_data_norm_trans.values[:, :, 1:] = np.arcsinh(cell_data_norm_trans[:, :, 1:])


        #cell_data.to_netcdf(os.path.join(save_dir, point, 'segmented_data.nc'))
        #cell_data_norm.to_netcdf(os.path.join(save_dir, point, 'segmented_data_normalized.nc'))
        #cell_data_norm_trans.to_netcdf(os.path.join(save_dir, point + '_segmented_data_normalized_transformed.nc'))

        csv_format = pd.DataFrame(data=cell_data_norm_trans.values[0, :, :], columns=cell_data.features)
        combined = pd.concat([csv_format, cell_props], axis=1)
        combined.to_csv(os.path.join(save_dir, point + "_normalized_transformed.csv"), index=False)

        csv_format = pd.DataFrame(data=cell_data_norm.values[0, :, :], columns=cell_data.features)
        combined = pd.concat([csv_format, cell_props], axis=1)
        combined.to_csv(os.path.join(save_dir, point + "_normalized.csv"), index=False)


def concatenate_csv(base_dir, csv_files, column_name="point", column_values=None):
    """Take a list of CSV paths and concatenates them together, adding in the identifier in column_values
    
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
        raise ValueError("csv_files and column_values have different lengths: csv {}, column_values {}".format(len(csv_files), len(column_values)))

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

    combined_data.to_csv(os.path.join(base_dir, "combined_data.csv"))
