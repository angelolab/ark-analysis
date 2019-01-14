import numpy as np
import os
import pandas as pd
import copy
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.stats as stats


os.chdir("/Users/noahgreenwald/Documents/Grad School/Lab/Segmentation/Contours/First_Run/Figs/")

# first attempt to evaluate accuracy of different networks by comparing to gold standard contoured data

# read in TIFs containing ground truth contoured data, along with predicted segmentation
image_direc = '/Users/noahgreenwald/Documents/Grad School/Lab/Segmentation/Contours/First_Run/Point23/'
# predicted_data = plt.imread(image_direc + "Deep_Segmentation_Interior.tif")
predicted_data = plt.imread(image_direc + "Deep_Segmentation_Border.tif")
contour_data = plt.imread(image_direc + "Nuclear_Interior_Mask.tif")
contour_data.setflags(write=1)


# generates labels (L) for each distinct object in the image, along with their indices
# For some reason, the regionprops output is 0 indexed, such that the 1st cell appears at index 0.
# However, the labeling of these cells is 1-indexed, so that the 1st cell is given the label of 1
# It's hard to describe how dumb and confusing this is.
# Cell 452, for example, can have its regionprops data output by accessing the 451st index

predicted_L, predicted_idx = skimage.measure.label(predicted_data, return_num=True, connectivity=1)
predicted_props = skimage.measure.regionprops(predicted_L)

contour_L, contour_idx = skimage.measure.label(contour_data,return_num=True, connectivity=1)
contour_props = skimage.measure.regionprops(contour_L)


# remove labels from contour data that appear in padded region
# set padding
pad = 0
row = 0
while pad == 0:
    if np.sum(predicted_data[row, :] > 0):
        pad = row
    else:
        row += 1

pad_mask = np.zeros((1024, 1024), dtype="bool")
pad_mask[0:30, :] = True
pad_mask[:, 0:30] = True
pad_mask[:, -30:-1] = True
pad_mask[-30:-1, :] = True
remove_ids = np.unique(contour_L[pad_mask])
remove_idx = np.isin(contour_L, remove_ids)
contour_L[remove_idx] = 0

contour_data[contour_L == 0] = 0

# regenerate object IDs after removing regions that overlap with padding
contour_L, contour_idx = skimage.measure.label(contour_data,return_num=True, connectivity=1)
contour_props = skimage.measure.regionprops(contour_L)


#  determine how well the contoured data was recapitulated by the predicted segmentaiton data
cell_frame = pd.DataFrame(columns=["contour_cell", "contour_cell_size", "predicted_cell", "predicted_cell_size",
                                   "percent_overlap", "merged", "split", "missing"], dtype="float")

cell_frame_interior = copy.deepcopy(cell_frame)

for contour_cell in range(1, contour_idx + 1):

    # generate a mask for the contoured cell, get all predicted cells that overlap the mask
    mask = contour_L == contour_cell
    overlap_id, overlap_count = np.unique(predicted_L[mask], return_counts=True)
    overlap_id, overlap_count = np.array(overlap_id), np.array(overlap_count)

    # remove cells that aren't at least 5% of current cell
    contour_cell_size = sum(sum(mask))
    idx = overlap_count > 0.05 * contour_cell_size
    overlap_id, overlap_count = overlap_id[idx], overlap_count[idx]

    # sort the overlap counts in decreasing order
    sort_idx = np.argsort(-overlap_count)
    overlap_id, overlap_count = overlap_id[sort_idx], overlap_count[sort_idx]

    # go through logic to determine relationship between overlapping cells

    if overlap_count[0] / contour_cell_size > 0.9:

        # if greater than 90% of pixels contained in first overlap, assign to that cell
        pred_cell = overlap_id[0]
        pred_cell_size = predicted_props[pred_cell - 1].area
        split = False
        percnt = overlap_count[0] / contour_cell_size
        # TODO determine if this cell was merged with other contoured cells into larger predicted cell

        cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                        "predicted_cell": pred_cell, "predicted_cell_size": pred_cell_size,
                                        "percent_overlap": percnt, "merged": False, "split": False,
                                        "missing": False}, ignore_index=True)
    else:

        # Determine whether any other predicted cells contribute primarily to this contoured cell

        # Identify number of cells needed to get to 80% of total volume of cell
        cum_size = overlap_count[0]
        idx = 0
        while (cum_size / contour_cell_size) < 0.8:
            idx += 1
            cum_size += overlap_count[idx]

            if idx > 20:
                raise Exception("Something failed in the while loop")

        # Figure out which of these cells have at least 80% of their volume contained in original cell
        split_flag = False
        for cell in range(1, idx + 1):
            pred_cell_size = predicted_props[overlap_id[cell] - 1].area
            percnt = overlap_count[cell] / contour_cell_size
            if overlap_count[cell] / pred_cell_size > 0.7 and overlap_id[cell] != 0:

                split_flag = True
                cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                "predicted_cell": overlap_id[cell], "predicted_cell_size": pred_cell_size,
                                                "percent_overlap": percnt, "merged": False, "split": True,
                                                "missing": False}, ignore_index=True)
            else:
                # this cell hasn't been split, just poorly assigned
                pass

        # assign first cell, based on whether or not subsequent cells indicate split
        cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                        "predicted_cell": overlap_id[0], "predicted_cell_size": overlap_count[0],
                                        "percent_overlap": overlap_count[0] / contour_cell_size, "merged": False,
                                        "split": split_flag, "missing": False}, ignore_index=True)


def outline_objects(L_matrix, list_of_lists):
    """takes in an L matrix generated by skimage.label, along with a list of lists, and returns a mask that has each
    list of cells highlighted in a different color for plotting"""

    L_plot = copy.deepcopy(L_matrix)
    for idx, list in enumerate(list_of_lists):

        mask = np.isin(L_plot, list)
        L_plot[mask] = idx + 2

    L_plot[L_plot > idx + 2] = 1

    return(L_plot)

# determine if any of the contoured cells were merged together into a larger cell
cell_frame = cell_frame[cell_frame["contour_cell_size"] > 10]
missing_idx = cell_frame["predicted_cell"] == 0.
merge_idx = np.logical_and(cell_frame["predicted_cell"].duplicated(), ~missing_idx)
cell_frame.loc[merge_idx, "merged"] = True
cell_frame.loc[missing_idx, "missing"] = True

plt.scatter(cell_frame["contour_cell_size"], cell_frame["predicted_cell_size"])
plt.xlabel("Contoured Cell")
plt.ylabel("Predicted Cell")

split_cells = cell_frame.loc[cell_frame["split"] == 1, "predicted_cell"]
split_cells = [x for x in split_cells if x != 0]
merged_cells = cell_frame.loc[cell_frame["merged"] == 1, "predicted_cell"]
merged_cells = [x for x in merged_cells if x != 0]
bad_cells = cell_frame.loc[cell_frame["percent_overlap"] < 0.8, "predicted_cell"]

bad_cells = [x for x in bad_cells if x != 0]
x = outline_objects(predicted_L, [split_cells, merged_cells, bad_cells])


fig, ax = plt.subplots()

cmap = mpl.colors.ListedColormap(['Black', 'Grey', 'Blue', 'Pink', 'Yellow'])
# set limits .5 outside true range
mat = ax.imshow(x,cmap=cmap,vmin = np.min(x)-.5, vmax = np.max(x)+.5)
#tell the colorbar to tick at integers
cbar = fig.colorbar(mat, ticks=np.arange(np.min(x),np.max(x)+1))
cbar.ax.set_yticklabels(['first', 'second', 'third', 'fourth', 'fifth'])





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




seq1 = np.arange(0,10,1)
seq2 = np.arange(10,15,0.5)

example1 = pd.DataFrame(np.random.randn(10,3), columns=["One", "Two","Three"])




# identify cutoff values. Cells without issues are plotted in dark blue, small in light green, big in yellow
# if ratio of contoured / deep is greater than 1, that means the deep output is too small, and cells have been
# hyper segmented. If ratio is less than 1, that means deep ouput is too big, and cells have been merged together


# calculate percent of background pixels in true image that are included in cells in image 2
cell_frame['base_in_background'] = 0
cell_frame['mapped_in_background'] = 0

for idx, cell in enumerate(cell_frame['base_cell']):
    mask = contour_L == cell
    target = deep_L[mask]
    cell_frame.loc[idx, 'base_in_background'] = sum(target == 0) / np.sum(mask)


my_fig = plt.figure()
plt.hist(cell_frame['base_in_background'][cell_frame['base_cell'] > 1])
plt.xlabel("Percentage of contoured cell in background in deep map")
my_fig.savefig(plot_path + "Figure_4.pdf")


# this likely puts the value in the wrong row, but since using it for histogram only doesn't matter
for idx, cell in enumerate(cell_frame['mapped_cell'][cell_frame['mapped_cell'] > 0]):
     mask = deep_L == cell
     target = contour_L[mask]
     cell_frame.loc[idx, 'mapped_in_background'] = sum(target == 0) / np.sum(mask)


my_fig = plt.figure()
plt.hist(cell_frame['mapped_in_background'][cell_frame['base_cell'] > 1])
plt.xlabel("Percentage of deep cells in background in contoured map")
my_fig.savefig(plot_path + "Figure_5.pdf")


# This is likely an artifact due to imdilate command being run on deep images, which makes them much bigger