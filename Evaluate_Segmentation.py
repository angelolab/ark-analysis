import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import skimage.io as io
import helper_functions
import importlib
import os
importlib.reload(helper_functions)


# code to evaluate accuracy of different segmentation contours

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
#base_dir = '/Users/noahgreenwald/Google Drive/Grad School/Lab/Segmentation_Contours/Practice_Run_Zips/'

deep_direc = base_dir + 'analyses/20190703_watershed_comparison/'
plot_direc = deep_direc + 'figs/'

files = os.listdir(deep_direc)
files = [file for file in files if 'mask' in file]

file_name = "3_class_64_filters_256_densefilters_epoch_30_mask_python_max_python.tiff"


for file in files:
    file_name = file

    predicted_data = io.imread(deep_direc + file_name)

    #contour_data = io.imread(base_dir + "Point23/Nuclear_Interior_Mask.tif")
    contour_data = io.imread(base_dir + "Point23/Nuclear_Mask_Label.tif")

    cell_frame, predicted_label, contour_label = helper_functions.compare_contours(predicted_data, contour_data)

    # plot_overlay(base_dir, np.zeros((1024, 1024)), predicted_label, contour_label)


    # remove small objects, but not zero sized, as these are missing errors
    cell_frame = cell_frame[cell_frame["contour_cell_size"] > 10]
    cell_frame = cell_frame[np.logical_or(cell_frame["predicted_cell_size"] > 10, cell_frame["predicted_cell_size"] == 0)]

    # find predicted cells which have been associated with multiple distinct ground truth cells, excluding background
    merge_idx = np.logical_and(cell_frame["predicted_cell"].duplicated(), np.logical_not(cell_frame["missing"]))
    merge_ids = cell_frame["predicted_cell"][merge_idx]

    # make sure both copies of predicted cell are marked as merged
    merge_idx_complete = np.isin(cell_frame["predicted_cell"], merge_ids)
    cell_frame.loc[merge_idx_complete, "merged"] = True


    # figure out which cells are labeled as both low_quality and merged
    # because merge classification comes only from see duplicates of predicted ID, low_quality is more accurate call
    # as it was determined by actual overlap pattern
    unmerge_ids = cell_frame.loc[np.logical_and(cell_frame["merged"], cell_frame["low_quality"]), "predicted_cell"]
    unmerge_idx = np.isin(cell_frame["predicted_cell"], unmerge_ids)
    cell_frame.loc[unmerge_idx, "merged"] = False

    # figure out which cells are labeled as both split and merged, meaning messy segmentation which is low quality
    double_merge = cell_frame.loc[np.logical_and(cell_frame["merged"], cell_frame["split"]), "predicted_cell"]
    double_split = cell_frame.loc[np.logical_and(cell_frame["merged"], cell_frame["split"]), "contour_cell"]
    double_merge_idx = np.isin(cell_frame["predicted_cell"], double_merge)
    double_split_idx = np.isin(cell_frame["contour_cell"], double_split)
    cell_frame.loc[np.logical_or(double_merge_idx, double_split_idx), ["merged", "split", "low_quality"]] = [False, False, True]

    # check to make sure no double counting
    np.sum(np.logical_and(cell_frame["merged"], cell_frame["split"]))
    np.sum(np.logical_and(cell_frame["merged"], cell_frame["low_quality"]))
    np.sum(np.logical_and(cell_frame["low_quality"], cell_frame["split"]))

    split_cells = cell_frame.loc[cell_frame["split"], "predicted_cell"]
    merged_cells = cell_frame.loc[cell_frame["merged"], "predicted_cell"]
    bad_cells = cell_frame.loc[cell_frame["low_quality"], "predicted_cell"]
    missing_cells = cell_frame.loc[cell_frame["missing"], "contour_cell"]
    created_cells = cell_frame.loc[cell_frame["created"], "predicted_cell"]
    print(len(missing_cells))
    print(len(created_cells))

    # create list of cells that don't have any of the above
    accurate_cells = cell_frame["predicted_cell"]
    error_cells = split_cells.append(merged_cells).append(bad_cells).append(created_cells)
    accurate_idx = np.isin(accurate_cells, error_cells)
    accurate_cells = accurate_cells[~accurate_idx]

    # plot error analysis
    swapped = helper_functions.randomize_labels(copy.copy(contour_label))
    classify_outline = helper_functions.outline_objects(predicted_label, [split_cells, merged_cells, bad_cells])

    helper_functions.plot_color_map(classify_outline, ground_truth=None, save_path=None)

    # make subplots for two simple plots

    helper_functions.plot_barchart_errors(cell_frame)

# mean average precision
new_iou = helper_functions.calc_iou_matrix(contour_label, predicted_label)

iou_thresholds = np.arange(0.5, 1, 0.05)
scores, false_negatives, false_positives = helper_functions.calc_modified_average_precision(new_iou, iou_thresholds)
np.mean(scores)

# get cell_ids for cells which don't pass iou_threshold
mAP_errors = np.where(false_positives[7, :])[0]

# increment by one to adjust for 0-based indexing in iou function, then remove those that have known error
mAP_errors = np.arange(1, 2) + mAP_errors
mAP_errors = mAP_errors[np.isin(mAP_errors, accurate_cells)]

# plot errors
classify_outline = helper_functions.outline_objects(predicted_label, [mAP_errors, error_cells])

helper_functions.plot_color_map(classify_outline, names=['Background', 'Normal Cell', 'mAP Errors', 'Segmentation Errors'])

plot_overlay(base_dir, np.zeros((1024, 1024)), predicted_label, contour_label)


# plot values for cells with more than 1 neighbor vs those without
adj_mtrx = helper_functions.calc_adjacency_matrix(predicted_label)
neighbor_num = np.sum(adj_mtrx, axis=0)
many_neighbors = np.where(neighbor_num > 2)


split_cells = np.asarray(split_cells)
merged_cells = np.asarray(merged_cells)
bad_cells = np.asarray(bad_cells)


classify_outline = helper_functions.outline_objects(predicted_label, [split_cells[np.isin(split_cells, many_neighbors)],
                                                                      merged_cells[np.isin(merged_cells, many_neighbors)],
                                                                      bad_cells[np.isin(bad_cells, many_neighbors)]])
helper_functions.plot_color_map(classify_outline, ground_truth=None, save_path=None)



classify_outline = helper_functions.outline_objects(predicted_label, [split_cells[~np.isin(split_cells, many_neighbors)],
                                                                      merged_cells[~np.isin(merged_cells, many_neighbors)],
                                                                      bad_cells[~np.isin(bad_cells, many_neighbors)]])
helper_functions.plot_color_map(classify_outline, ground_truth=None, save_path=None)
