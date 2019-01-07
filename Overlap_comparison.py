# Let's get things started

import numpy as np
import pandas as pd
import copy
import skimage.measure
import matplotlib.pyplot as plt
import scipy


# using matploblib to read in TIF data

image_direc = '/Users/noahgreenwald/Documents/Grad School/Lab/Segmentation/SegmentationSamir/'

deep_data = plt.imread(image_direc + "patient2deepCellSegmentationInterior.tif")
ilastic_data = plt.imread(image_direc + "point2IlastikSegmentationInterior.tif")
contour_data = plt.imread(image_direc + "patient2interior.tif")


# number each of the labels, then get size of each
deep_L, deep_idx = skimage.measure.label(deep_data,return_num=True, connectivity=1)
deep_props = skimage.measure.regionprops(deep_L)

contour_L, contour_idx = skimage.measure.label(contour_data,return_num=True, connectivity=1)
contour_props = skimage.measure.regionprops(contour_L)

ilastic_L, ilastic_idx = skimage.measure.label(ilastic_data, return_num=True, connectivity=1)
ilastic_props = skimage.measure.regionprops(ilastic_L)


cell_frame = pd.DataFrame(columns=["base_cell", "base_cell_size", "mapped_cell", "mapped_cell_size", "ratio"], dtype="float")

# iterate through each cell, get cells that it overlaps, and add to db
tar_idx = deep_idx
tar_L = deep_L
tar_props = deep_props

base_idx = contour_idx
base_L = contour_L
base_props = contour_props
for base_cell in range(1, base_idx + 1):
    # generate a mask for the location of current cell, get all cells that overlap the mask in target cell
    #if base_props[base_cell - 1].area == 1:
    #    pass
    #else:
        mask = base_L == base_cell
        overlap_id, overlap_count = np.unique(tar_L[mask], return_counts=True)
        overlap_id, overlap_count = np.array(overlap_id), np.array(overlap_count)

        # remove cells that aren't at least 5% of current cell
        idx = overlap_count > 0.05*sum(overlap_count)
        overlap_id, overlap_count = overlap_id[idx], overlap_count[idx]

        for pos, ele in enumerate(overlap_id):
            # if overlaps with 0 (background), include only if that's the only overlap
            if ele == 0:
                if len(overlap_id) == 1:
                    cell_frame = cell_frame.append(
                        {"base_cell": base_cell, "base_cell_size": base_props[base_cell - 1].area,
                         "mapped_cell": ele, "mapped_cell_size": 0}, ignore_index=True)
            else:
                cell_frame = cell_frame.append({"base_cell":base_cell, "base_cell_size":base_props[base_cell - 1].area,
                                            "mapped_cell":ele, "mapped_cell_size":tar_props[ele - 1].area}, ignore_index=True)



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


def outline_objects(L_matrix, small_list, big_list):
    "takes in an L matrix generated by skimage.label, and returns a version with only cells in quesetion highlighted"
    L_frame = pd.DataFrame(L_matrix).copy()
    mask_small = L_frame.isin(small_list)
    mask_big = L_frame.isin(big_list)
    L_frame[L_frame > 0] = 1
    L_frame[mask_small] = 2
    L_frame[mask_big] = 3
    return(L_frame)


# identify cutoff values. Cells without issues are plotted in dark blue, small in light green, big in yellow
# if ratio of contoured / deep is greater than 1, that means the deep output is too small, and cells have been
# hyper segmented. If ratio is less than 1, that means deep ouput is too big, and cells have been merged together

cell_frame_small = cell_frame.copy()
cell_frame_small = cell_frame_small[cell_frame_small['base_cell_size'] > 1]
cell_frame_small = cell_frame_small[cell_frame_small['mapped_cell_size'] != 0]

small_ids = cell_frame_small['mapped_cell'][cell_frame_small['ratio'] > 1.5].values
big_ids = cell_frame_small['mapped_cell'][cell_frame_small['ratio'] < 0.3].values
temp = outline_objects(deep_L, small_ids, big_ids)

my_fig = plt.figure(dpi=1000)
plt.imshow(temp)
plot_path = "/Users/noahgreenwald/Documents/Grad School/Lab/Segmentation/Plots/"
my_fig.savefig(plot_path + "Figure_3.pdf")

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