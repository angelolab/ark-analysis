# helper functions for everything else

import os
import numpy as np
import copy
import skimage.morphology as morph
import skimage.measure
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import skimage.io as io
import skimage.filters.rank as rank
import scipy.ndimage as nd
from skimage.segmentation import find_boundaries


# data loading
def save_deepcell_tifs(model_output_xr, save_path, transform='pixel', points=None, watershed_smooth=3, pixel_smooth=5):
    """Extract and save tifs from deepcell output and save in directory format

        Args
            model_output: xarray of tifs output by deepcell
            save_path: folder to save tifs
            transform: one of pixel, fgbg, watershed, which determines how to process/save image
            points: optional list of points to extract. If none extracts all
            watershed_smooth: side length for square selem used for median smoothing
            pixel_smooth: variance used for gaussian filter smoothing
            """

    if len(model_output_xr.shape) != 4:
        raise ValueError("xarray data has the wrong dimensions, expecting 4")

    if points is None:
        points = model_output_xr.coords['points']
    else:
        if np.any(~np.isin(points, model_output_xr.coords['points'])):
            raise ValueError("Incorrect list of points given, not all are present in data structure")

    # keep only the selected points
    if len(points) == 1:
        # don't subset, will change dimensions
        pass
    else:
        model_output_xr = model_output_xr.loc[points, :, :, :]

    if transform == 'watershed':
        if model_output_xr.shape[-1] != 4:
            raise ValueError("Watershed transform selected, but last dimension is not 4")
        if model_output_xr.coords['masks'].values[0] != 'level_0':
            raise ValueError("Watershed transform selected, but first channel is not Level_0")

        # find the max value across different energy levels within each point
        argmax_images = []
        for j in range(model_output_xr.shape[0]):
            argmax_images.append(np.argmax(model_output_xr[j, ...].values, axis=-1))
        argmax_images = np.array(argmax_images)

        # create array to hold argmax and smoothed argmax mask
        watershed_processed = np.zeros(argmax_images.shape + (2, ))
        watershed_processed[:, :, :, 0] = argmax_images

        for i in range(model_output_xr.shape[0]):
            smoothed_argmax = rank.median(argmax_images[i, ...], np.ones((watershed_smooth, watershed_smooth)))
            watershed_processed[i, :, :, 1] = smoothed_argmax

            io.imsave(os.path.join(save_path, model_output_xr.name + '_' + model_output_xr.coords['points'].values[i] +
                                   '_watershed.tiff'), watershed_processed[i, :, :, 0].astype('int16'))

            io.imsave(os.path.join(save_path, model_output_xr.name + '_' + model_output_xr.coords['points'].values[i] +
                                   '_watershed_smoothed.tiff'), watershed_processed[i, :, :, 1].astype('int16'))

        mask = ["watershed", "watershed_smoothed"]
        watershed_processed_xr = xr.DataArray(watershed_processed, name=model_output_xr.name + '_processed',
                                           coords=[model_output_xr.coords['points'].values, range(1024), range(1024), mask],
                                           dims=["points", "rows", "cols", "masks"])
        watershed_processed_xr.to_netcdf(save_path + '/' + watershed_processed_xr.name + '.nc')

    elif transform == 'pixel':
        if model_output_xr.shape[-1] != 3:
            raise ValueError("pixel transform selected, but last dimension is not three")
        if model_output_xr.coords['masks'].values[0] != 'border':
            raise ValueError("pixel transform selected, but mask names don't match")

        pixel_processed = np.zeros(model_output_xr.shape[:-1] + (3, ))
        pixel_processed[:, :, :, 0:2] = model_output_xr.loc[:, :, :, ['border', 'interior']].values

        for i in range(model_output_xr.shape[0]):
            # smooth interior probability for each point
            smoothed_int = nd.gaussian_filter(model_output_xr[i, :, :, 1], pixel_smooth)
            pixel_processed[i, :, :, 2] = smoothed_int

            # save tifs
            io.imsave(os.path.join(save_path, model_output_xr.name + '_' + model_output_xr.coords['points'].values[i] +
                                   '_pixel_border.tiff'), pixel_processed[i, :, :, 0].astype('float32'))
            io.imsave(os.path.join(save_path, model_output_xr.name + '_' + model_output_xr.coords['points'].values[i] +
                                   '_pixel_interior.tiff'), pixel_processed[i, :, :, 1].astype('float32'))
            io.imsave(os.path.join(save_path, model_output_xr.name + '_' + model_output_xr.coords['points'].values[i] +
                                   '_pixel_interior_smoothed.tiff'), pixel_processed[i, :, :, 2].astype('float32'))

        # save output
        mask_labels = ["pixel_border", "pixel_interior", "pixel_interior_smoothed"]
        pixel_processed_xr = xr.DataArray(pixel_processed, name=model_output_xr.name + '_processed',
                                            coords=[model_output_xr.coords['points'], range(1024), range(1024), mask_labels],
                                            dims=["points", "rows", "cols", "masks"])
        pixel_processed_xr.to_netcdf(save_path + '/' + pixel_processed_xr.name + '.nc')

    elif transform == 'fgbg':
        if model_output_xr.shape[-1] != 2:
            raise ValueError("fgbg flag set to false, but 2-level output not provided")

        deepcell_outputs = np.zeros(model_output_xr.shape[:-1] + (1, ))

        for i in range(model_output_xr.shape[0]):
            # smooth fgbg probability then threshold thresholding
            # thresholded_prob = model_output_xr[i, :, :, 1] > 0.7
            # deepcell_outputs[i, :, :, 0] = thresholded_prob
            deepcell_outputs[i, :, :, 0] = model_output_xr[i, :, :, 1]

            if cohort:
                # save files in different folders
                if not os.path.exists(os.path.join(save_path, file_names[i])):
                    os.makedirs(os.path.join(save_path, file_names[i]))

                io.imsave(os.path.join(save_path, file_names[i], '_fgbg.tiff'),
                          deepcell_outputs[i, :, :, 0].astype('float32'))


            else:
                # save files in same folder
                io.imsave(os.path.join(save_path, file_names[i] + '_fgbg.tiff'),
                          deepcell_outputs[i, :, :, 0].astype('float32'))


def load_tifs_from_points_dir(point_dir, tif_folder, points=None, tifs=None):
    """Takes a set of TIFs from a directory structure organised by points, and loads them into a numpy array.

        Args:
            point_dir: directory path to points
            tif_folder: name of tif_folder within each point
            points: optional list of point_dirs to load, otherwise loads all folders with Point in name
            tifs: optional list of TIFs to load, otherwise loads all TIFs

        Returns:
            Numpy array with shape [points, tifs, x_dim, y_dim]
    """

    if not os.path.isdir(point_dir):
        raise ValueError("Directory does not exist")

    if points is None:
        # get all point folders
        points = os.listdir(point_dir)
        points = [point for point in points if 'Point' in point]
        points = [point for point in points if os.path.isdir(os.path.join(point_dir, point))]
    else:
        # use supplied list, but check to make sure they all exist
        for point in points:
            if not os.path.isdir(os.path.join(point_dir, point)):
                raise ValueError("Could not find point folder {}".format(point))

    if len(points) == 0:
        raise ValueError("No points found in directory")

    if not os.path.isdir(os.path.join(point_dir, points[0], tif_folder)):
        raise ValueError("Invalid tif folder name")

    # get tifs from first point directory if no tif names supplied
    if tifs is None:
        tifs = os.listdir(os.path.join(point_dir, points[0], tif_folder))
        tifs = [tif for tif in tifs if '.tif' in tif]

    if len(tifs) == 0:
        raise ValueError("No tifs found in designated folder")

    # check to make sure supplied tifs exist
    for tif in tifs:
        if not os.path.isfile(os.path.join(point_dir, points[0], tif_folder, tif)):
            raise ValueError("Could not find {} in supplied directory {}".format(tif, os.path.join(point_dir, points[0], tif_folder, tif)))

    test_img = io.imread(os.path.join(point_dir, points[0], tif_folder, tifs[0]))
    img_data = np.zeros((len(points), len(tifs), test_img.shape[0], test_img.shape[1]))

    for point in range(len(points)):
        for tif in range(len(tifs)):
            img_data[point, tif, :, :] = io.imread(os.path.join(point_dir, points[point], tif_folder, tifs[tif]))

    img_xr = xr.DataArray(img_data, coords=[points, tifs, range(test_img.shape[0]), range(test_img.shape[0])],
                          dims=["point", "channel", "rows", "cols"])

    return img_xr


def segment_images(input_images, segmentation_masks):
    """Extract single cell protein expression data from channel TIFs for a single point

        Args:
            input_images (xarray): Channels x TIFs matrix of imaging data
            segmentation_masks (numpy array): mask_type x mask matrix of segmentation data

        Returns:
            xr_counts: xarray containing segmented data of cells x markers"""

    if type(input_images) is not xr.DataArray:
        raise ValueError("Incorrect data type for ground_truth, expecting xarray")

    if type(segmentation_masks) is not xr.DataArray:
        raise ValueError("Incorrect data type for masks, expecting xarray")

    if input_images.shape[1:] != segmentation_masks.shape[1:]:
        raise ValueError("Image data and segmentation masks have different dimensions")

    max_cell_num = np.max(segmentation_masks[0, :, :].values).astype('int')

    # create np.array to hold subcellular_loc x channel x cell info
    cell_counts = np.zeros((segmentation_masks.shape[0], max_cell_num + 1, len(input_images.channel) + 1))

    # loop through each segmentation mask
    for subcell_loc in range(segmentation_masks.shape[0]):
        # for each mask, loop through each cell to figure out marker count
        for cell in range(1, max_cell_num + 1):

            # get mask corresponding to current cell
            cell_mask = segmentation_masks[subcell_loc, :, :] == cell
            cell_size = np.sum(cell_mask)

            # calculate the total signal intensity within that cell mask across all channels, and save to numpy
            channel_counts = np.sum(input_images.values[:, cell_mask], axis=1)
            cell_counts[subcell_loc, cell, 1:] = channel_counts

            cell_counts[subcell_loc, cell, 0] = cell_size

    # create xarray  to hold resulting data
    col_names = np.concatenate((np.array('cell_size'), input_images.channel), axis=None)
    xr_counts = xr.DataArray(cell_counts, coords=[segmentation_masks.subcell_loc, range(max_cell_num + 1), col_names],
                             dims=['subcell_loc', 'cell_id', 'cell_data'])
    return xr_counts


# plotting functions

def plot_overlay(predicted_contour, plotting_tif=None, alternate_contour=None, path=None):
    """Take in labeled contour data, along with optional mibi tif and second contour, and overlay them for comparison"

    Args:
        predicted_contour: 2D numpy array of labeled cell objects
        plotting_tif: 2D numpy array of imaging signal
        alternate_contour: 2D numpy array of labeled cell objects
        path: path to save the resulting image

    outputs:
        plot viewer: plots the outline(s) of the mask(s) as well as intensity from plotting tif
            predicted_contour in red
            alternate_contour in white
        overlay: saves as TIF in file path if specified
    """

    if plotting_tif.shape != predicted_contour.shape:
        raise ValueError("plotting_tif and predicted_contour array dimensions not equal.")

    if len(np.unique(predicted_contour)) < 2:
        raise ValueError("predicted contour is not labeled")

    if path is not None:
        if os.path.exists(os.path.split(path)[0]) is False:
            raise ValueError("File path does not exist.")

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(predicted_contour, connectivity=1, mode='inner').astype(np.uint8)

    # creates transparent mask for easier visualization of TIF data
    rgb_mask = np.ma.masked_where(predicted_contour_mask == 0, predicted_contour_mask)

    if alternate_contour is not None:

        if predicted_contour.shape != alternate_contour.shape:
            raise ValueError("predicted_contour and alternate_contour array dimensions not equal.")

        # define borders of cell in mask
        alternate_contour_mask = find_boundaries(alternate_contour, connectivity=1, mode='inner').astype(np.uint8)

        # creates transparent mask for easier visualization of TIF data
        rgb_mask_2 = np.ma.masked_where(alternate_contour_mask == 0, predicted_contour_mask)

        # creates plots overlaying ground truth and predicted contour masks
        overlay = plt.figure()
        plt.imshow(plotting_tif, clim=(0, 15))
        plt.imshow(rgb_mask_2, cmap="Greys", interpolation='none')
        plt.imshow(rgb_mask, cmap='autumn', interpolation='none')

        if path is not None:
            overlay.savefig(os.path.join(path), dpi=800)

    else:
        # if only one mask provided
        overlay = plt.figure()
        plt.imshow(plotting_tif, clim=(0, 15))
        plt.imshow(rgb_mask, cmap='autumn', interpolation='none')

        if path is not None:
            overlay.savefig(os.path.join(path), dpi=800)


def randomize_labels(label_map):
    """Takes in a labeled matrix and swaps the integers around so that color gradient has better contrast

    Inputs:
    label_map(2D numpy array): labeled TIF with each object assigned a unique value

    Outputs:
    swapped_map(2D numpy array): labeled TIF with object labels permuted"""

    max_val = np.max(label_map)
    for cell_target in range(1, max_val):
        swap_1 = cell_target
        swap_2 = np.random.randint(1, max_val)
        swap_1_mask = label_map == swap_1
        swap_2_mask = label_map == swap_2
        label_map[swap_1_mask] = swap_2
        label_map[swap_2_mask] = swap_1

    label_map = label_map.astype('int16')

    return label_map


def outline_objects(L_matrix, list_of_lists):
    """takes in an L matrix generated by skimage.label, along with a list of lists, and returns a mask that has the
    pixels for all cells from each list represented as integer values for easy plotting"""

    L_plot = copy.deepcopy(L_matrix).astype(float)

    for idx, val in enumerate(list_of_lists):
        mask = np.isin(L_plot, val)

        # use a negative value to not interfere with cell labels
        L_plot[mask] = -(idx + 2)

    L_plot[L_plot > 1] = 1
    L_plot = np.absolute(L_plot)
    L_plot = L_plot.astype('int16')
    return L_plot


def plot_color_map(outline_matrix, names,
                   plotting_colors=['Black', 'Grey', 'Blue', 'Green', 'Pink', 'moccasin', 'tan', 'sienna', 'firebrick'],
                   ground_truth=None, save_path=None):
    """Plot label map with cells of specified category colored the same

        Args
            outline_matrix: output of outline_objects function which assigns same value to cells of same class
            names: list of names for each category to use for plotting
            ground truth: optional argument to supply label map of true segmentation to be plotted alongside
            save_path: optional argument to save plot as TIF

        Returns
            Displays plot in window"""

    # TODO: add option to supply color palette

    num_categories = np.max(outline_matrix)
    plotting_colors = plotting_colors[:num_categories + 1]
    cmap = mpl.colors.ListedColormap(plotting_colors)

    if ground_truth is not None:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        mat = ax[0].imshow(outline_matrix, cmap=cmap, vmin=np.min(outline_matrix) - .5,
                           vmax=np.max(outline_matrix) + .5)
        swapped = helper_functions.randomize_labels(ground_truth)
        ax[1].imshow(swapped)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        mat = ax.imshow(outline_matrix, cmap=cmap, vmin=np.min(outline_matrix) - .5,
                           vmax=np.max(outline_matrix) + .5)

    # tell the colorbar to tick at integers
    cbar = fig.colorbar(mat, ticks=np.arange(np.min(outline_matrix), np.max(outline_matrix) + 1))

    cbar.ax.set_yticklabels(names)


    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)


def plot_barchart_errors(pd_array, cell_category=["split", "merged", "low_quality"], save_path=None):

    """Plot different error types in a barchart, along with cell-size correlation in a scatter plot
        Args
            pd_array: pandas cell array representing error types for each class of cell
            cell_category: list of error types to extract from array
            save_path: optional file path to save generated TIF

        Returns
            Display plot on viewer"""

    # make sure all supplied categories are column names
    if np.any(~np.isin(cell_category, pd_array.columns)):
        raise ValueError("Invalid column name")

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].scatter(pd_array["contour_cell_size"], pd_array["predicted_cell_size"])
    ax[0].set_xlabel("Contoured Cell")
    ax[0].set_ylabel("Predicted Cell")

    # compute percentage of different error types
    errors = np.zeros(len(cell_category))
    for i in range(len(errors)):
        errors[i] = len(set(pd_array.loc[pd_array[cell_category[i]], "predicted_cell"]))

    errors = errors / len(set(pd_array["predicted_cell"]))
    position = range(len(errors))
    ax[1].bar(position, errors)

    ax[1].set_xticks(position)
    ax[1].set_xticklabels(cell_category)
    ax[1].set_title("Fraction of cells misclassified")

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

def plot_barchart(values, labels, title, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    position = range(len(values))
    ax.bar(position, values)
    ax.set_xticks(position)
    ax.set_xticklabels(labels)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)


# training data generation
def process_training_data(interior_contour, interior_border_contour):
    """Take in a contoured map of the border of each cell as well as entire cell, and generate annotated label map
    where each cell is its own unique pixel value

    Args:
        interior_contour: TIF with interior pixels as 1s, all else as 0
        interior_border_contour: TIF with all cellular pixels as 1s, all else as 0s

    Returns:
        label_contour: np.array with pixels belonging to each cell as a unique integer"""

    if np.sum(interior_contour) == 0:
        raise ValueError("Border contour is empty array")

    if np.sum(interior_border_contour) == 0:
        raise ValueError("Cell contour is empty array")

    if np.sum(interior_contour) > np.sum(interior_border_contour):
        raise ValueError("Arguments are likely switched, interior_contour is larger than interior_border_contour")

    # label cells
    interior_contour = skimage.measure.label(interior_contour, connectivity=1)

    # for each individual cell, expand to slightly larger than original size one pixel at a time
    new_masks = copy.copy(interior_contour)
    for idx in range(2):
        for cell_label in np.unique(new_masks)[1:]:
            img = new_masks == cell_label
            img = morph.binary_dilation(img, morph.square(3))
            new_masks[img] = cell_label

    # set pixels to 0 anywhere outside bounds of original shape
    label_contour = randomize_labels(new_masks.astype("int"))
    label_contour[interior_border_contour == 0] = 0

    missed_pixels = np.sum(np.logical_and(interior_border_contour > 0, label_contour < 1))
    print("New Erosion and dilating resulted in a total of {} pixels "
          "that are no longer marked out of {}".format(missed_pixels, interior_contour.shape[0] ** 2))

    return label_contour


# accuracy evaluation
def compare_contours(predicted_label, contour_label):

    """Compares two distinct segmentation outputs

    Args:
        predicted_label: label map generated by algorithm
        contour_label: label map generated from ground truth data

    Returns:
        cell_frame: a pandas dataframe containing metrics for each cell in ground truth data"""

    # check to see if data has been supplied with labels already, or needs to be labeled
    if len(np.unique(predicted_label)) < 3:
        predicted_label = skimage.measure.label(predicted_label, connectivity=1)

    if len(np.unique(contour_label)) < 3:
        contour_label = skimage.measure.label(contour_label, connectivity=1)

    # get region props of predicted cells and initialize datastructure for storing values
    cell_frame = pd.DataFrame(columns=["contour_cell", "contour_cell_size", "predicted_cell", "predicted_cell_size",
                                       "percent_overlap", "merged", "split", "missing", "low_quality", "created"])

    # loop through each contoured cell, and compute accuracy metrics for overlapping predicting cells
    for contour_cell in range(1, np.max(contour_label) + 1):
        # generate a mask for the contoured cell, get all predicted cells that overlap the mask
        mask = contour_label == contour_cell
        if np.sum(mask) < 15:
            print("found another small cell {}".format(contour_cell))
            continue

        overlap_id, overlap_count = np.unique(predicted_label[mask], return_counts=True)
        overlap_id, overlap_count = np.array(overlap_id), np.array(overlap_count)

        # remove cells that aren't at least 5% of current cell
        contour_cell_size = np.sum(mask)
        idx = overlap_count > 0.05 * contour_cell_size
        overlap_id, overlap_count = overlap_id[idx], overlap_count[idx]

        # sort the overlap counts in decreasing order
        sort_idx = np.argsort(-overlap_count)
        overlap_id, overlap_count = overlap_id[sort_idx], overlap_count[sort_idx]

        # check and see if maps primarily to background
        if overlap_id[0] == 0:
            if overlap_count[0] / contour_cell_size > 0.8:
                # more than 80% of cell is overlapping with background, classify predicted cell as missing
                cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                "predicted_cell": 0, "predicted_cell_size": 0,
                                                "percent_overlap": overlap_count[0] / contour_cell_size, "merged": False,
                                                "split": False, "missing": True, "low_quality": False,
                                                "created": False}, ignore_index=True)
                continue
            else:
                # not missing, just bad segmentation. Classify predicted cell as bad
                # TODO: figure out how often this condition is true, what do we do with remaining overlap targets
                cell_frame = cell_frame.append(
                    {"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                     "predicted_cell": overlap_id[1], "predicted_cell_size": np.sum(predicted_label == overlap_id[1]),
                     "percent_overlap": overlap_count[0] / contour_cell_size, "merged": False,
                     "split": False, "missing": False, "low_quality": True, "created": False}, ignore_index=True)
                continue
        else:
            # remove background as target cell and change cell size to for calculation
            if 0 in overlap_id:
                keep_idx = overlap_id != 0
                contour_cell_size -= overlap_count[~keep_idx][0]
                overlap_id, overlap_count = overlap_id[keep_idx], overlap_count[keep_idx]

        # go through logic to determine relationship between overlapping cells
        # TODO: change logic to include a too small category for when cell is completely contained within but is smaller
        if overlap_count[0] / contour_cell_size > 0.9:

            # if greater than 90% of pixels contained in first overlap, assign to that cell
            pred_cell = overlap_id[0]
            pred_cell_size = np.sum(predicted_label == pred_cell)
            percnt = overlap_count[0] / contour_cell_size

            cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                            "predicted_cell": pred_cell, "predicted_cell_size": pred_cell_size,
                                            "percent_overlap": percnt, "merged": False, "split": False,
                                            "missing": False, "low_quality": False, "created": False}, ignore_index=True)
        else:
            # No single predicted cell occupies more than 90% of contour cell size, figure out the type of error made
            split_flag = False
            bad_flag = False
            # TODO check if first cell also has at least 80% of volume contained in contour cell?
            # TODO can keep a counter of number of cells that meet this criteria, if >2 then split?

            # keep only cells that overlap at least 20% with target cell
            idx = overlap_count > 0.2 * contour_cell_size
            overlap_id, overlap_count = overlap_id[idx], overlap_count[idx]
            for cell in range(1, len(overlap_id)):
                pred_cell_size = np.sum(predicted_label == overlap_id[cell])
                percnt = overlap_count[cell] / contour_cell_size
                if overlap_count[cell] / pred_cell_size > 0.7:
                    # multiple predicted cells were assigned to single target cell, hence split
                    split_flag = True
                    cell_frame = cell_frame.append(
                        {"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                         "predicted_cell": overlap_id[cell], "predicted_cell_size": pred_cell_size,
                         "percent_overlap": percnt, "merged": False, "split": True,
                         "missing": False, "low_quality": False, "created": False}, ignore_index=True)
                else:
                    # this cell hasn't been split, just poorly assigned
                    bad_flag = True
                    cell_frame = cell_frame.append(
                        {"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                         "predicted_cell": overlap_id[cell], "predicted_cell_size": pred_cell_size,
                         "percent_overlap": percnt, "merged": False, "split": False,
                         "missing": False, "low_quality": True, "created": False}, ignore_index=True)

            # assign the first cell, based on whether or not subsequent cells indicate split or bad
            if bad_flag and split_flag:
                bad_flag = False
            cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                            "predicted_cell": overlap_id[0], "predicted_cell_size": overlap_count[0],
                                            "percent_overlap": overlap_count[0] / contour_cell_size, "merged": False,
                                            "split": split_flag, "missing": False, "low_quality": bad_flag,
                                            "created": False}, ignore_index=True)

    # check and see if any new cells were created in predicted_label that don't exist in contour_label
    for predicted_cell in range(1, np.max(predicted_label) + 1):
        if not np.isin(predicted_cell, cell_frame["predicted_cell"]):
            cell_frame = cell_frame.append({"contour_cell": 0, "contour_cell_size": 0, "predicted_cell": predicted_cell,
                                            "predicted_cell_size": np.sum(predicted_label == predicted_cell),
                                            "percent_overlap": 0, "merged": False, "split": split_flag,
                                            "missing": False, "low_quality": False, "created": True}, ignore_index=True)

    return cell_frame, predicted_label, contour_label

# DSB-score adapted from https://www.biorxiv.org/content/10.1101/580605v1.full
# object IoU matrix adapted from code written by Morgan Schwartz in deepcell-tf/metrics

def calc_iou_matrix(ground_truth_label, predicted_label):
    """Calculates pairwise ious between all cells from two masks

    Args:
        ground_truth_label: 2D label array representing ground truth contours
        predicted_label: 2D labeled array representing predicted contours

    Returns:
        iou_matrix: matrix of ground_truth x predicted cells with iou value for each
    """

    if len(np.unique(ground_truth_label)) < 3:
        raise ValueError("Ground truth array was not pre-labeled")

    if len(np.unique(predicted_label)) < 3:
        raise ValueError("Predicted array was not pre-labeled")

    iou_matrix = np.zeros((np.max(ground_truth_label), np.max(predicted_label)))

    for i in range(1, iou_matrix.shape[0] + 1):
        gt_img = ground_truth_label == i
        overlaps = np.unique(predicted_label[gt_img])
        for j in overlaps:
            pd_img = predicted_label == j
            intersect = np.sum(np.logical_and(gt_img, pd_img))
            union = np.sum(np.logical_or(gt_img, pd_img))

            # adjust index by one to account for not including background
            iou_matrix[i - 1, j - 1] = intersect / union
    return iou_matrix


def calc_modified_average_precision(iou_matrix, thresholds):
    """Calculates the average precision between two masks across a range of iou thresholds

    Args:
        iou_matrix: intersection over union matrix created by calc_iou_matrix function
        thresholds: list used to threshold iou values in matrix

    Returns:
        scores: list of modified average precision values for each threshold
        false_neg_idx: array of booleans indicating whether cell was flagged as false positive at each threshold
        false_pos_idx: array of booleans indicating whether cell was flagged as false negative at each threshold"""

    if np.max(iou_matrix) > 1:
        raise ValueError("Improperly formatted iou_matrix, contains values greater than 1")

    if len(thresholds) == 0:
        raise ValueError("Must supply at least one threshold value")

    if np.any(np.logical_or(thresholds > 1, thresholds < 0)):
        raise ValueError("Thresholds must be between 0 and 1")

    scores = []
    false_neg_idx = np.zeros((len(thresholds), iou_matrix.shape[0]))
    false_pos_idx = np.zeros((len(thresholds), iou_matrix.shape[1]))

    for i in range(len(thresholds)):

        # threshold iou_matrix as designated value
        iou_matrix_thresh = iou_matrix > thresholds[i]

        # Calculate values based on projecting along prediction axis
        pred_proj = iou_matrix_thresh.sum(axis=1)

        # Zeros (aka absence of hits) correspond to true cells missed by prediction
        false_neg = pred_proj == 0
        false_neg_idx[i, :] = false_neg
        false_neg = np.sum(false_neg)

        # Calculate values based on projecting along truth axis
        truth_proj = iou_matrix_thresh.sum(axis=0)

        # Empty hits indicate predicted cells that do not exist in true cells
        false_pos = truth_proj == 0
        false_pos_idx[i, :] = false_pos
        false_pos = np.sum(false_pos)

        # Ones are true positives
        true_pos = np.sum(pred_proj == 1)

        score = true_pos / (true_pos + false_pos + false_neg)
        scores.append(score)

    return scores, false_neg_idx, false_pos_idx


def calc_adjacency_matrix(label_map, border_dist=0):
    """Generate matrix describing which cells are within the specified distance from one another

    Args
        label map: numpy array with each distinct cell labeled with a different pixel value
        border_dist: number of pixels separating borders of adjacent cells in order to be classified as neighbors

    Returns
        adjacency_matrix: numpy array of num_cells x num_cells with a 1 for neighbors and 0 otherwise"""

    if len(np.unique(label_map)) < 3:
        raise ValueError("array must be provided in labeled format")

    if not isinstance(border_dist, int):
        raise ValueError("Border distance must be an integer")

    adjacency_matrix = np.zeros((np.max(label_map) + 1, np.max(label_map) + 1), dtype='int')

    # We need to expand enough pixels such that cells which are within the specified border distance will overlap.
    # To find cells that are 0 pixels away, we expand 1 pixel in each direction to find overlaps
    # To check for cells that are 1 pixel away, we expand 2 pixels in either direction
    # we also need to factor in a center pixel which adds a constant of one
    morph_dist = border_dist * 2 + 3

    for cell in range(1, np.max(label_map) + 1):
        mask = label_map == cell
        mask = morph.dilation(mask, morph.square(morph_dist))
        overlaps = np.unique(label_map[mask])
        adjacency_matrix[cell, overlaps] = 1

    # reset background distance to 0
    adjacency_matrix[0, :] = 0
    adjacency_matrix[:, 0] = 0

    return adjacency_matrix

def euc_dist(coords_1, coords_2):
    """Calculate the euclidian distance between two y,x tuples

        Args
            coords_1: tuple of row, col values
            coords_2: tuple of row, col values

        Returns
            dist: distance between two points"""

    y = coords_1[0] - coords_2[0]
    x = coords_1[1] - coords_2[1]
    dist = math.sqrt(x**2 + y**2)
    return dist

def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells

        Args
            label_map: numpy array with unique cells given unique pixel labels

        Returns
            dist_matrix: cells x cells matrix with the euclidian distance between centers of corresponding cells"""

    if len(np.unique(label_map)) < 3:
        raise ValueError("Array must be provided in labeled format")

    cell_num = np.max(label_map)
    dist_matrix = np.zeros((cell_num + 1, cell_num + 1))
    props = skimage.measure.regionprops(label_map)

    for cell in range(1, cell_num + 1):
        cell_coords = props[cell - 1].centroid
        for tar_cell in range(1, cell_num + 1):
            tar_coords = props[tar_cell - 1].centroid
            dist = euc_dist(cell_coords, tar_coords)
            dist_matrix[cell, tar_cell] = dist

    return dist_matrix

