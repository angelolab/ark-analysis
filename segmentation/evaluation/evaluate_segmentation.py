import numpy as np
import skimage.io as io
import os
import copy

import pandas as pd
import xarray as xr

import importlib
from segmentation.utils import segmentation_utils, plot_utils, evaluation_utils

importlib.reload(plot_utils)


# code to evaluate accuracy of different segmentation contours

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200327_Metrics_Comparison/'

data_dir = os.path.join(base_dir, 'model_output')
plot_dir = os.path.join(base_dir, 'figs')

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

predicted_labels = io.imread(os.path.join(data_dir, 'predicted_labels.tiff'))

true_labels = io.imread(os.path.join(data_dir, 'true_labels.tiff'))

cell_frame, predicted_label, contour_label = evaluation_utils.compare_contours(predicted_labels, true_labels)

# read in ground truth annotations to create overlays
input_xr = xr.load_dataarray(os.path.join(data_dir, "deepcell_input.xr"))

plot_utils.plot_overlay(predicted_labels, input_xr[0, :, :, :].values, true_labels)
                              #os.path.join(plot_direc, file_name + '_overlay_border.tiff'))


# remove all small ground truth objects
cell_frame = cell_frame[cell_frame["contour_cell_size"] > 5]

# remove small predicted objects, but keep those equal to 0, since these are misses
cell_frame = cell_frame[np.logical_or(cell_frame["predicted_cell_size"] > 5, cell_frame["predicted_cell_size"] == 0)]

# find predicted cells which have been associated with multiple distinct ground truth cells, excluding background
merge_idx = np.logical_and(cell_frame["predicted_cell"].duplicated(), np.logical_not(cell_frame["missing"]))
merge_ids = cell_frame["predicted_cell"][merge_idx]

# make sure both copies of predicted cell are marked as merged
merge_idx_complete = np.isin(cell_frame["predicted_cell"], merge_ids)
cell_frame.loc[merge_idx_complete, "merged"] = True


# figure out which cells are labeled as both low_quality and merged
# The merge classification comes only from seen duplicates of predicted ID. Therefore, if the ID is duplicated
# because it got flagged during low quality logic, low_quality is the more accurate classification

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
print("there are {} total cells counted as merged and split".format(np.sum(np.logical_and(cell_frame["merged"], cell_frame["split"]))))
print("there are {} cells counted as merged and lq".format(np.sum(np.logical_and(cell_frame["merged"], cell_frame["low_quality"]))))
print("there are {} cells counted as lq and split".format(np.sum(np.logical_and(cell_frame["low_quality"], cell_frame["split"]))))

split_cells = cell_frame.loc[cell_frame["split"], "predicted_cell"]
merged_cells = cell_frame.loc[cell_frame["merged"], "predicted_cell"]
bad_cells = cell_frame.loc[cell_frame["low_quality"], "predicted_cell"]
missing_cells = cell_frame.loc[cell_frame["missing"], "contour_cell"]
created_cells = cell_frame.loc[cell_frame["created"], "predicted_cell"]

print("there are {} total cells counted as missing".format(len(missing_cells)))
print("ther are {} total cells counted as created ".format(len(created_cells)))


# create gradations of low quality cells
bad_cells_80 = cell_frame.loc[np.logical_and(cell_frame["percent_overlap"] > .8, cell_frame["percent_overlap"] < 0.9), "predicted_cell"]
bad_cells_70 = cell_frame.loc[np.logical_and(cell_frame["percent_overlap"] > .7, cell_frame["percent_overlap"] < 0.8), "predicted_cell"]
bad_cells_60 = cell_frame.loc[np.logical_and(cell_frame["percent_overlap"] > .6, cell_frame["percent_overlap"] < 0.7), "predicted_cell"]

bad_cells_80 = [cell for cell in bad_cells_80 if cell in bad_cells and cell != 0]
bad_cells_70 = [cell for cell in bad_cells_70 if cell in bad_cells]
bad_cells_60 = [cell for cell in bad_cells_60 if cell in bad_cells]
bad_cells_50 = [x for x in bad_cells if x not in bad_cells_80 and x not in bad_cells_70 and x not in bad_cells_60]

# create list of cells that don't have any of the above
accurate_cells = cell_frame["predicted_cell"]
error_cells = split_cells.append(merged_cells).append(bad_cells).append(created_cells)
accurate_idx = np.isin(accurate_cells, error_cells)
accurate_cells = accurate_cells[~accurate_idx]

plotting_label = copy.copy(predicted_label)
cell_num = np.max(predicted_label) + 1
missing_cells_new = list(range(cell_num, cell_num + len(missing_cells) + 1))
for i in range(len(missing_cells)):
    cell_mask = contour_label == missing_cells.values[i]
    plotting_label[cell_mask] = cell_num
    cell_num += 1

# plot error analysis
swapped = plot_utils.randomize_labels(copy.copy(contour_label))
classify_outline = plot_utils.outline_objects(predicted_label, [split_cells, merged_cells, bad_cells])
plot_utils.plot_color_map(classify_outline, ["background", "fine", "split", "merged", "bad"], ground_truth=None)


classify_outline = plot_utils.outline_objects(plotting_label, [split_cells, merged_cells, missing_cells_new,
                                                                      bad_cells_80,
                                                                      bad_cells_70, bad_cells_60, bad_cells_50])

plot_utils.plot_color_map(classify_outline, ground_truth=None,
                                names=['Bg', 'Norm', 'split', 'merg', 'missing', '80', '70', '60', 'rest'])

io.imsave(os.path.join(plot_direc, file_name + 'color_map_raw.tiff'), classify_outline)
# make subplots for two simple plots

plot_utils.plot_barchart_errors(cell_frame, predicted_errors=['split', 'merged', 'low_quality'],
                                      contour_errors=["missing"])
# mean average precision
iou_matrix = segmentation_utils.calc_iou_matrix(contour_label, predicted_label)

iou_thresholds = np.arange(0.5, 1, 0.05)
scores, false_negatives, false_positives = segmentation_utils.calc_modified_average_precision(iou_matrix, iou_thresholds)
scores = scores + [np.mean(scores)]
plot_utils.plot_barchart(scores, np.concatenate(((iou_thresholds * 100).astype('int').astype('str'), ['average'])),
                               'IOU Errors', save_path=os.path.join(plot_direc, file_name + '_iou.tiff'))

# save pd dataframe for later loading of
iou_df = pd.DataFrame({'scores': scores, 'thresholds': iou_thresholds})
iou_df.to_pickle(plot_direc + 'iou_dataframe.pkl')

# optionally save accuracy metrics pandas array for future loading
cell_frame.to_pickle(plot_direc + "dataframe.pkl")

# deepcell metrics evaluation
# sys.path.append(os.path.abspath('../deepcell-tf'))
# from deepcell import metrics


# get cell_ids for cells which don't pass iou_threshold
mAP_errors = np.where(false_positives[7, :])[0]

# increment by one to adjust for 0-based indexing in iou function, then remove those that have known error
mAP_errors = np.arange(1, 2) + mAP_errors
mAP_errors = mAP_errors[np.isin(mAP_errors, accurate_cells)]

# plot errors
classify_outline = plot_utils.outline_objects(predicted_label, [mAP_errors, error_cells])

plot_utils.plot_color_map(classify_outline, names=['Background', 'Normal Cell', 'mAP Errors', 'Segmentation Errors'])

plot_utils.plot_overlay(base_dir, np.zeros((1024, 1024)), predicted_label, contour_label)


# plot values for cells with more than 1 neighbor vs those without
adj_mtrx = segmentation_utils.calc_adjacency_matrix(predicted_label)
neighbor_num = np.sum(adj_mtrx, axis=0)
many_neighbors = np.where(neighbor_num > 2)


split_cells = np.asarray(split_cells)
merged_cells = np.asarray(merged_cells)
bad_cells = np.asarray(bad_cells)


classify_outline = plot_utils.outline_objects(predicted_label, [split_cells[np.isin(split_cells, many_neighbors)],
                                                                      merged_cells[np.isin(merged_cells, many_neighbors)],
                                                                      bad_cells[np.isin(bad_cells, many_neighbors)]])
plot_utils.plot_color_map(classify_outline, ground_truth=None, save_path=None)



classify_outline = plot_utils.outline_objects(predicted_label, [split_cells[~np.isin(split_cells, many_neighbors)],
                                                                      merged_cells[~np.isin(merged_cells, many_neighbors)],
                                                                      bad_cells[~np.isin(bad_cells, many_neighbors)]])
plot_utils.plot_color_map(classify_outline, ground_truth=None, save_path=None)
