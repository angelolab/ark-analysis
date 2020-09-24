import os
import copy

import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity


# plotting functions

def plot_overlay(predicted_contour, plotting_tif, alternate_contour=None, path=None):
    """Take in labeled contour data, along with optional mibi tif and second contour,
    and overlay them for comparison"

    Plots the outline(s) of the mask(s) as well as intensity from plotting tif.  Predicted
    contours are plotted in red, while alternate contours are plotted in white.  Plot is
    saved as a TIF file in provided file path, if specified.

    Args:
        predicted_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal
        alternate_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        path (str):
            path to save the resulting image
    """

    if plotting_tif is None:
        # will just plot the outlines
        pass
    else:
        if len(plotting_tif.shape) == 2:
            if plotting_tif.shape != predicted_contour.shape:
                raise ValueError("plotting_tif and predicted_contour array dimensions not equal.")
            else:
                # convert RGB image with same data across all three channels
                plotting_tif = np.stack((plotting_tif, plotting_tif, plotting_tif), axis=2)
        elif len(plotting_tif.shape) == 3:
            blank_channel = np.zeros(plotting_tif.shape[:2] + (1,), dtype=plotting_tif.dtype)
            if plotting_tif.shape[2] == 1:
                # pad two empty channels
                plotting_tif = np.concatenate((plotting_tif, blank_channel, blank_channel), axis=2)
            elif plotting_tif.shape[2] == 2:
                # pad one empty channel
                plotting_tif = np.concatenate((plotting_tif, blank_channel), axis=2)
            elif plotting_tif.shape[2] == 3:
                # don't need to do anything
                pass
            else:
                raise ValueError("only 3 channels of overlay supported, got {}".
                                 format(plotting_tif.shape))
        else:
            raise ValueError("plotting tif must be 2D or 3D array, got {}".
                             format(plotting_tif.shape))

    if path is not None:
        if os.path.exists(os.path.split(path)[0]) is False:
            raise ValueError("File path does not exist.")

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(predicted_contour,
                                             connectivity=1, mode='inner').astype(np.uint8)
    predicted_contour_mask[predicted_contour_mask > 0] = 255

    if plotting_tif is None:
        # will just save the contour mask
        io.imsave(path, predicted_contour_mask)
    else:
        # rescale each channel to go from 0 to 255
        rescaled = np.zeros(plotting_tif.shape, dtype='uint8')

        for idx in range(plotting_tif.shape[2]):
            if np.max(plotting_tif[:, :, idx]) == 0:
                # don't need to rescale this channel
                pass
            else:
                percentiles = np.percentile(plotting_tif[:, :, idx][plotting_tif[:, :, idx] > 0],
                                            [5, 95])
                rescaled_intensity = rescale_intensity(plotting_tif[:, :, idx],
                                                       in_range=(percentiles[0], percentiles[1]),
                                                       out_range='uint8')
                rescaled[:, :, idx] = rescaled_intensity

        # overlay first contour on all three RGB, to have it show up as white border
        rescaled[predicted_contour_mask > 0, :] = 255

        # overlay second contour as red outline if present
        if alternate_contour is not None:

            if predicted_contour.shape != alternate_contour.shape:
                raise ValueError("predicted_contour and alternate_"
                                 "contour array dimensions not equal.")

            # define borders of cell in mask
            alternate_contour_mask = find_boundaries(alternate_contour, connectivity=1,
                                                     mode='inner').astype(np.uint8)
            rescaled[alternate_contour_mask > 0, 0] = 255
            rescaled[alternate_contour_mask > 0, 1:] = 0

        # save as TIF if path supplied, otherwise display on screen
        if path is not None:
            io.imsave(path, rescaled)
        else:
            io.imshow(rescaled)


def randomize_labels(label_map):
    """Takes in a labeled matrix and swaps the integers around
    so that color gradient has better contrast

    Args:
        label_map (numpy.ndarray): labeled TIF with each object assigned a unique value

    Returns:
        numpy.ndarray:
            2D array corresponding to a labeled TIF with permuted object labels"""

    unique_vals = np.unique(label_map)[1:]
    pos_1 = np.random.choice(unique_vals, size=len(unique_vals))
    pos_2 = np.random.choice(unique_vals, size=len(unique_vals))

    for i in range(len(pos_1)):
        swap_1 = pos_1[i]
        swap_2 = pos_2[i]
        swap_1_mask = label_map == swap_1
        swap_2_mask = label_map == swap_2
        label_map[swap_1_mask] = swap_2
        label_map[swap_2_mask] = swap_1

    label_map = label_map.astype('int16')

    return label_map


# TODO: make documentation more specific here
def outline_objects(L_matrix, list_of_lists):
    """Takes in an L matrix generated by skimage.label, along with a
    list of lists, and returns a mask that has the
    pixels for all cells from each list represented as integer values for easy plotting

    Args:
        L_matrix (numpy.ndarray):
            a label map indicating the label of each cell
        list_of_lists (list):
            each element is a list of cells we wish to plot separately

    Returns:
        np.ndarray:
            an binary mask indicating the regions of cells outlined
    """

    L_plot = copy.deepcopy(L_matrix).astype(float)

    for idx, val in enumerate(list_of_lists):
        mask = np.isin(L_plot, val)

        # use a negative value to not interfere with cell labels
        L_plot[mask] = -(idx + 2)

    L_plot[L_plot > 1] = 1
    L_plot = np.absolute(L_plot)
    L_plot = L_plot.astype('int16')
    return L_plot


def plot_color_map(outline_matrix, names, plotting_colors=None, ground_truth=None, save_path=None):
    """Plot label map with cells of specified category colored the same

    Displays plot in window

    Args:
        outline_matrix (numpy.ndarray):
            output of outline_objects function which assigns same value to cells of same class
        names (list):
            list of names for each category to use for plotting
        plotting_colors (list):
            list of colors to use for plotting cell categories
        ground_truth (numpy.ndarray):
            optional argument to supply label map of true segmentation to be plotted alongside
        save_path (str):
            optional argument to save plot as TIF
    """

    if plotting_colors is None:
        plotting_colors = ['Black', 'Grey', 'Blue', 'Green',
                           'Pink', 'moccasin', 'tan', 'sienna', 'firebrick']

    num_categories = np.max(outline_matrix)
    plotting_colors = plotting_colors[:num_categories + 1]
    cmap = mpl.colors.ListedColormap(plotting_colors)

    if ground_truth is not None:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        mat = ax[0].imshow(outline_matrix, cmap=cmap, vmin=np.min(outline_matrix) - .5,
                           vmax=np.max(outline_matrix) + .5)
        swapped = randomize_labels(ground_truth)
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


# TODO: make documentation more specific here
def plot_barchart_errors(pd_array, contour_errors, predicted_errors, save_path=None):
    """Plot different error types in a barchart, along with cell-size correlation in a scatter plot

    Args:
        pd_array (pandas.array):
            pandas cell array representing error types for each class of cell
        contour_errors (list):
            list of contour error types to extract from array
        predicted_errors (list):
            list of predictive error types to extract from the array
        save_path (str):
            optional file path to save generated TIF
    """

    # make sure all supplied categories are column names
    if np.any(~np.isin(contour_errors + predicted_errors, pd_array.columns)):
        raise ValueError("Invalid column name")

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].scatter(pd_array["contour_cell_size"], pd_array["predicted_cell_size"])
    ax[0].set_xlabel("Contoured Cell")
    ax[0].set_ylabel("Predicted Cell")

    # compute percentage of different error types
    errors = np.zeros(len(predicted_errors) + len(contour_errors))
    for i in range(len(contour_errors)):
        errors[i] = len(set(pd_array.loc[pd_array[contour_errors[i]], "contour_cell"]))

    for i in range(len(predicted_errors)):
        errors[i + len(contour_errors)] = len(set(pd_array.loc[pd_array[predicted_errors[i]],
                                                               "predicted_cell"]))

    errors = errors / len(set(pd_array["predicted_cell"]))
    position = range(len(errors))
    ax[1].bar(position, errors)

    ax[1].set_xticks(position)
    ax[1].set_xticklabels(contour_errors + predicted_errors)
    ax[1].set_title("Fraction of cells misclassified")

    if save_path is not None:
        fig.savefig(save_path, dpi=200)


def plot_mod_ap(mod_ap_list, thresholds, labels):
    df = pd.DataFrame({'iou': thresholds})

    for idx, label in enumerate(labels):
        df[label] = mod_ap_list[idx]['scores']

    fig, ax = plt.subplots()
    for label in labels:
        ax.plot('iou', label, data=df, linestyle='-', marker='o')

    ax.set_xlabel('IOU Threshold')
    ax.set_ylabel('mAP')
    ax.legend()
    fig.show()


def plot_error_types(errors, labels, error_plotting):
    data_dict = pd.DataFrame(pd.Series(errors[0])).transpose()

    for i in range(1, len(labels)):
        data_dict = data_dict.append(errors[i], ignore_index=True)

    data_dict['algos'] = labels

    fig, axes = plt.subplots(len(error_plotting))
    for i in range(len(error_plotting)):
        barchart_helper(ax=axes[i], values=data_dict[error_plotting[i]], labels=labels,
                        title='{} Errors'.format(error_plotting[i]))

    fig.show()
    fig.tight_layout()


def barchart_helper(ax, values, labels, title):
    positions = range(len(values))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
