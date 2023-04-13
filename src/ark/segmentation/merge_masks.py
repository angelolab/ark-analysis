# Merge masks functionality
# combine ez or other objects with MESMER or other cell masks.

import numpy as np
import pandas as pd
from skimage.io import imsave, imread
from skimage.measure import regionprops_table

# Function: merge_masks
# Combines overlapping object and cell soma masks, e.g. microglial projections and microglial cell soma, captured by
# different segmentation strategies. Cell masks are compared to combined masks; for any combination which represents at
# least the input percent of cell overlap (e.g. 70%), the combined mask is kept and incorporated into the original
# object masks to generate a new set of masks.
#
# calculates a mask for circular or 'blob' like objects, e.g. single large cells, or amyloid plaques. Uses a system of
# blurring the image, thresholding the blurred data using either a given number or using adaptive thresholding, removes
# small holes using the same input or adaptive input, and removing too small or too large objects before returning the
# object masks.
# input image = tif to segment on, object_shape_type = specify whether the object is more blob or projection shaped,
# blur = the sigma value for the gaussian blur, thresh = apply a global threshold for image thresholding if desired,
# hole_size = apply a specific size of area to close small holes over in object masks,
# fov_dim = the length, in um, of one side of the captured area within the tif,
# pix_dim = the number of pixels of one dimension of the tif,
# minpix = the minimum size of an object to capture, maxpix = the maximum size of an object to capture

# returns the cells remaining mask, which will be used for the next cycle in merging while there are objects and cells
# to merge. When no more cells and objects are left to merge, the final, non-merged cells are returned.

def merge_masks(object_mask, cell_mask, overlap, object_name, mask_save_path):
    # combine object and cell images using 'and' producing a binary overlap image
    binary_overlap_image = np.bitwise_and(object_mask, cell_mask)
    binary_overlap_image = binary_overlap_image > 0
    # label the cell mask image
    cell_mask_labels = skimage.measure.label(cell_mask)
    # using regionprops_table, input binary overlap image as intensity data and cell label image as label image
    find_cells_table = regionprops_table(label_image=cell_mask_labels, intensity_image=binary_overlap_image,
                                         properties=['label', 'intensity_mean'])
    # convert table into pandas data frame
    find_cells_df = pd.DataFrame(find_cells_table)
    # any cell properties with mean intensity over the threshold of positivity (per_overlap), keep. Discard the rest.
    cells_to_merge = find_cells_df[find_cells_df.intensity_mean > overlap / 100]
    # Create image with only the filtered overlapping cells
    cells_to_merge_mask = np.isin(cell_mask_labels, cells_to_merge['label'])
    # combine images into one and relabel. Save image with new labels as named object + cell mask merges.
    final_overlap_image = np.bitwise_and(object_mask, cells_to_merge_mask)
    imsave(fname=str([mask_save_path, "/", object_name, "_merged.tif"]), arr=final_overlap_image)
    # save cell masks without the masks that have been incorporated into the merged selections
    cells_to_keep_mask = np.isin(cell_mask_labels, cells_to_merge['label'], invert=True)

    return cells_to_keep_mask

# Function: merge_masks_seq
# Sequentially merge object masks with cell masks. Object list order is enforced, e.g. object_list[0] will merge
# overlapping object masks with cell masks from the initial cell segmentation. Remaining, un-merged cell masks will then
# be used to merge with object_list[1], etc.
#
# object_list = a list of names representing previously generated object masks. Need to be in order of merge status
#   e.g. microglia projections merged first with cells, then astrocyte projections, etc
# object_mask_path = path to where the original object masks are located
# cell_mask_path = path to where the original cell mask is located
# overlap = percent overlap of total pixel area needed for an object to be merged to a cell
# save_path = where merged masks, remaining cell mask will be saved

# returns a confirmation of successful merge status
def merge_masks_seq(object_list, object_mask_path, cell_mask_path, overlap, save_path):
    curr_cell_mask = imread(fname=str(cell_mask_path, '/', 'cells.tif')) # or whatever the normal post-MESMER suffix is
    for obj in object_list:
        curr_object_mask = imread(fname=str(object_mask_path, '/', obj, '.tif'))
        remaining_cells = merge_masks(object_mask=curr_object_mask, cell_mask=curr_cell_mask, percent_overlap=overlap,
                                      object_name=obj, mask_save_path=save_path)
        curr_cell_mask = remaining_cells
    imsave(fname=str([save_path, "final_cells_remaining.tif"]), arr=curr_cell_mask)
    return 1

