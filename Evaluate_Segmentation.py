import numpy as np
import skimage.io as io
import helper_functions
import os
import copy

import importlib
importlib.reload(helper_functions)


# code to evaluate accuracy of different segmentation contours

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/'

deep_direc = base_dir + 'analyses/20190914_tuning/'
plot_direc = deep_direc + 'figs/'

files = os.listdir(deep_direc)
files = [file for file in files if 'Point8' in file]

file_name = "Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_processedpixel_expansion_7.tiff"
file_name = "Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_processedpoint1_5marker_watershed_label_mask.tiff"
file_name = "Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_3marker_processedpoint1_3marker_watershed_label_mask.tiff"
file_name = "Training_Freeze_1_Nuc_HH3_81_rf_512_dense_128_conv_epoch_18_processedpoint1_HH3_watershed_label_mask.tiff"

nuc_seg = True
melanoma_val = True

if nuc_seg:
    file_name = "Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_processed_label_mask.tiff"
else:
    file_name = "Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_processed_label_mask.tiff"

for file in files:
    file_name = file

    predicted_data = io.imread(deep_direc + file_name)

    if nuc_seg:
        if melanoma_val:
            contour_data = io.imread(base_dir + "20190813_combined_data/Zips/Point1_Nuc_Mask_Label.tif")
        else:
            contour_data = io.imread(base_dir + "20190823_TA489_Redo/zips/Point8_Nuc_Mask_Label.tif")
    else:
        if melanoma_val:
            contour_data = io.imread(base_dir + "20190813_combined_data/Zips/Point1_Cell_Mask_Label.tif")
        else:
            contour_data = io.imread(base_dir + "20190823_TA489_Redo/zips/Point8_Cell_Mask_Label.tif")

    cell_frame, predicted_label, contour_label = helper_functions.compare_contours(predicted_data, contour_data)

    # read in ground truth annotations to create overlays
    # TODO: read in _processed.nc file to pull from
    HH3 = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190823_TA489_Redo/Point8/TIFs/HH3.tif')
    HH3 = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/Point1/HH3.tif')
    interior = io.imread(deep_direc + 'Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_point8_pixel_interior_smoothed.tiff')
    #
    # if melanoma_val:
    #     if nuc_seg:
    #         border = io.imread(deep_direc + 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_point8_pixel_border.tiff')
    #     else:
    #         border = io.imread(deep_direc + 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_point8_pixel_border.tiff')
    # else:
    #     if nuc_seg:
    #         border = io.imread(deep_direc + 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_point8_pixel_border.tiff')
    #     else:
    #         border = io.imread(deep_direc + 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_point8_pixel_border.tiff')

    helper_functions.plot_overlay(predicted_data, HH3, contour_data,
                                  os.path.join(plot_direc, file_name + '_overlay_border.tiff'))


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
    swapped = helper_functions.randomize_labels(copy.copy(contour_label))
    classify_outline = helper_functions.outline_objects(predicted_label, [split_cells, merged_cells, bad_cells])
    helper_functions.plot_color_map(classify_outline, ground_truth=None,
                                    save_path=os.path.join(plot_direc, file_name + '_color_map.tiff'))


    classify_outline = helper_functions.outline_objects(plotting_label, [split_cells, merged_cells, missing_cells_new,
                                                                          bad_cells_80,
                                                                          bad_cells_70, bad_cells_60, bad_cells_50])

    helper_functions.plot_color_map(classify_outline, ground_truth=None,
                                    names=['Bg', 'Norm', 'split', 'merg', 'missing', '80', '70', '60', 'rest'],
                                    save_path=os.path.join(plot_direc, file_name + '_color_map.tiff'))

    io.imsave(os.path.join(plot_direc, file_name + 'color_map_raw.tiff'), classify_outline)
    # make subplots for two simple plots

    helper_functions.plot_barchart_errors(cell_frame, predicted_errors=['split', 'merged', 'low_quality'],
                                          contour_errors=["missing"],
                                          save_path=os.path.join(plot_direc, file_name + '_stats.tiff'))

    # mean average precision
    iou_matrix = helper_functions.calc_iou_matrix(contour_label, predicted_label)

    iou_thresholds = np.arange(0.5, 1, 0.05)
    scores, false_negatives, false_positives = helper_functions.calc_modified_average_precision(iou_matrix, iou_thresholds)
    scores = scores + [np.mean(scores)]
    helper_functions.plot_barchart(scores, np.concatenate(((iou_thresholds * 100).astype('int').astype('str'), ['average'])),
                                   'IOU Errors', save_path=os.path.join(plot_direc, file_name + '_iou.tiff'))

# deepcell metrics evaluation
sys.path.append(os.path.abspath('../deepcell-tf'))
from deepcell import metrics


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
