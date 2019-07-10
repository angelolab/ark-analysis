import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
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

file_name = "mask_3_class_64_filters_256_densefilters_epoch_30.tiff"


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


    swapped = helper_functions.randomize_labels(copy.copy(contour_label))
    classify_outline = helper_functions.outline_objects(predicted_label, [split_cells, merged_cells, bad_cells])

    fig, ax = plt.subplots(nrows=1, ncols=2)
    cmap = mpl.colors.ListedColormap(['Black', 'Grey', 'Blue', 'Red', 'Yellow'])

    # set limits .5 outside true range
    mat = ax[0].imshow(classify_outline, cmap=cmap, vmin=np.min(classify_outline)-.5, vmax=np.max(classify_outline)+.5)
    ax[1].imshow(swapped)

    # tell the colorbar to tick at integers
    cbar = fig.colorbar(mat, ticks=np.arange(np.min(classify_outline), np.max(classify_outline)+1))

    #cbar.ax.set_yticklabels(['Background', 'Normal', 'Split', 'Merged', 'Low Quality'])
    fig.tight_layout()

    fig.savefig(plot_direc + file_name + '_color_map.tiff', dpi=200)

    #io.imsave(plot_direc + file_base + file_suf + '_color_map_raw.tiff', classify_outline)
    #io.imsave(plot_direc + file_base + file_suf + '_ground_truth_raw.tiff', swapped * 100)

    # make subplots for two simple plots
    fig, ax = plt.subplots(2, 1, figsize=(10,10))

    ax[0].scatter(cell_frame["contour_cell_size"], cell_frame["predicted_cell_size"])
    ax[0].set_xlabel("Contoured Cell")
    ax[0].set_ylabel("Predicted Cell")

    # compute percentage of different error types
    errors = np.array([len(set(merged_cells)), len(set(split_cells)), len(set(bad_cells))]) / len(set(cell_frame["predicted_cell"]))
    position = range(len(errors))
    categories = ["Merged", "Split", "Bad"]
    ax[1].bar(position, errors)

    ax[1].set_xticks(position)
    ax[1].set_xticklabels(categories)
    ax[1].set_title("Fraction of cells misclassified")

    fig.savefig(plot_direc + file_name + '_stats.tiff', dpi=200)



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

fig, ax = plt.subplots(nrows=1, ncols=2)
cmap = mpl.colors.ListedColormap(['Black', 'Grey', 'Red', 'Yellow'])

# set limits .5 outside true range
mat = ax[0].imshow(classify_outline, cmap=cmap, vmin=np.min(classify_outline)-.5, vmax=np.max(classify_outline)+.5)
ax[1].imshow(swapped)
fig.tight_layout()

plot_overlay(base_dir, np.zeros((1024, 1024)), predicted_label, contour_label)


# plot values for the different classes of cells

# create matrix to keep track of number of neighbors each cell has
contour_stats = pd.DataFrame(np.zeros((contour_idx + 1, 2)))
contour_stats.columns = ["neighbors_5", "neighbors_7"]

contour_L_5 = scipy.ndimage.grey_dilation(contour_L, (5, 5))
contour_L_7 = scipy.ndimage.grey_dilation(contour_L, (7, 7))

# count number of times each cell overlaps with neighbor
for cell in range(1,contour_idx):
    overlap_ids = np.unique(contour_L_7[contour_L == cell])
    if len(overlap_ids) > 1:
        # don't count overlap with self
        idx = overlap_ids != cell
        overlap_ids = overlap_ids[idx]
        for overlap in overlap_ids:
            contour_stats.iloc[overlap, 1] += 1


cell_frame['ratio'] = cell_frame['base_cell_size'] / cell_frame['mapped_cell_size']

# loop through different neighbor values, plotting each one
neighbor_num = [-1, 0, 1, 2]
neighbor_labels = ["All cells", "No Adjacent Cells", "One Adjacent Cell", "2+ Adjacent Cells"]
plot_scatter = False
plt.figure()

for idx, val in enumerate(neighbor_num):
    cell_frame_small = cell_frame.copy()
    cell_frame_small = cell_frame_small[cell_frame_small['base_cell_size'] > 1]
    cell_frame_small = cell_frame_small[cell_frame_small['mapped_cell_size'] != 0]

    if idx == 0:
        plot_ids = contour_stats.index

    else:
        plot_ids = contour_stats[contour_stats['neighbors_7'] == val].index

    cell_frame_small = cell_frame_small[cell_frame_small['base_cell'].isin(plot_ids)]

    plt.subplot(2, 4, idx + 5)
    if plot_scatter:
        plt.scatter(cell_frame_small['base_cell_size'], (cell_frame_small['mapped_cell_size']), s=10)
    else:
        heatmap, xedges, yedges = np.histogram2d(cell_frame_small['base_cell_size'].astype(float),
                                                 cell_frame_small['mapped_cell_size'].astype(float), bins=80)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')

    plt.title(neighbor_labels[idx])

    if idx == 0:
        plt.ylabel("Cell Size: DeepCell")

    if idx == 2:
        plt.xlabel("Cell Size: Contoured")




cell_frame_small = cell_frame_small[cell_frame_small['ratio'] < 1.5]

heatmap, xedges, yedges = np.histogram2d(cell_frame_small['base_cell_size'].astype(float), cell_frame_small['mapped_cell_size'].astype(float), bins=80)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()