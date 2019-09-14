# swap the labels for cell objects from whole-cell to nucleus for a subset of the cells
# useful for changing messy annotations of whole cells to simpler annotations of just nucleus
import numpy as np
import copy
from skimage.segmentation import find_boundaries
import skimage.io as io
import matplotlib.pyplot as plt

points_list = ["Point4", "Point11", "Point12", "Point14", "Point15", "Point18", "Point19", "Point21"]

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/Zips/'
image_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/'

point = points_list[7]
for point in points_list:

    all_labels = io.imread(base_dir + point + '_Objects_Mask_Label.tif')
    nuc_labels = io.imread(base_dir + point + '_Nuc_Mask_Label.tif')

    vimentin = io.imread(image_dir + point + '/TIFs/Vimentin.tif')
    hlag = io.imread(image_dir + point + '/TIFs/HLAG.tif')

    switch_list = []

    # for each cell, if more than x% of pixels are positive for maker known to affect bad cells, swap it
    for cell in np.unique(all_labels):
        cell_mask = all_labels == cell
        vim_pos = np.sum(vimentin[cell_mask] > 0)
        hlag_pos = np.sum(hlag[cell_mask] > 0)

        if vim_pos / np.sum(cell_mask) > 0.35:
            pass

        elif hlag_pos / np.sum(cell_mask) > 0.35:
            pass
        else:
            switch_list.append(cell)

    # figure out which cells were flagged and remove their existing label
    flagged_cells = np.isin(all_labels, switch_list)
    modified_labels = copy.copy(all_labels)
    modified_labels[~flagged_cells] = 0
    io.imshow(modified_labels)

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(all_labels)
    ax[1].imshow(modified_labels)

    small_labels = copy.copy(all_labels)

    # figure out which nuclear label overlaps most with bad cell label, and replace with that one
    for cell in switch_list:
        cell_mask = all_labels == cell
        overlap_id, overlap_count = np.array(np.unique(nuc_labels[cell_mask], return_counts=True))

        if len(overlap_id) == 1:
            # do nothing, only overlaps with nuclear contour, no need to change
            pass
        else:
            if overlap_id[0] == 0:
                overlap_id = overlap_id[1:]
                overlap_count = overlap_count[1:]

            max = np.max(overlap_count)
            id = overlap_id[overlap_count == max]
            small_labels[cell_mask] = 0
            small_labels[nuc_labels == id] = cell

    io.imshow(small_labels)
    bounds = find_boundaries(small_labels, connectivity=1, mode='inner').astype('uint8')
    io.imsave(base_dir + point + '_Objects_Small_Nuclei_Outline.tiff', bounds)
    io.imsave(base_dir + point + '_Objects_Small_Nuclei_Mask.tiff', small_labels)
