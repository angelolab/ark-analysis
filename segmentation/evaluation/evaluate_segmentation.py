import numpy as np
import skimage.io as io
import os
import copy

import pandas as pd
import xarray as xr
import skimage

import importlib
from segmentation.utils import segmentation_utils, plot_utils, evaluation_utils

importlib.reload(evaluation_utils)


# code to evaluate accuracy of different segmentation contours

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200327_Metrics_Comparison/test_data/'

# data_dir = os.path.join(base_dir, 'model_output')
# plot_dir = os.path.join(base_dir, 'figs')

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

y_true = io.imread(base_dir + "true_labels.tiff")
y_pred = io.imread(base_dir + "predicted_labels.tiff")

# mean average precision
m_ap_array = evaluation_utils.compare_mAP({'y_true': [y_true], 'y_pred': [y_pred]})

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
