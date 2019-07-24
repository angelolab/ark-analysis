# take segmentation masks and TIFs and generate single cell data
import skimage.io as io
import skimage.morphology as morph
import skimage
import helper_functions
import importlib
import copy
import numpy as np
importlib.reload(helper_functions)


# load segmentation masks
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
seg_folder = ''
cell_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, ['Point23'], ['Nuclear_Mask_Label.tif'])
nuc_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, ['Point23'], ['Nuclear_Mask_Label.tif'])

# load TIFs
tif_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
tif_folder = 'TIFsNoNoise'
image_data = helper_functions.load_tifs_from_points_dir(tif_dir, tif_folder, ['Point23'], ['dsDNA.tif', 'LaminAC.tif'])

for i in range(len(image_data.channel)):
    point = image_data.point.values[i]

    # read in segmentation masks for whole-cell and nucleus
    cell_labels = cell_seg_data.loc[point, 'Nuclear_Mask_Label.tif', :, :].values.astype('int')

    # merge small cells?
    cell_props = skimage.measure.regionprops(cell_labels)

    nuc_mask = nuc_seg_data.loc[point, 'Nuclear_Mask_Label.tif', :, :].values.astype('int')

    # fake data for now
    nuc_mask = morph.erosion(nuc_mask, morph.square(3))
    cell_labels = morph.dilation(nuc_mask, morph.square(5))
    nuc_mask = nuc_mask > 0

    # duplicate whole cell data for cytoplasmic data
    cyto_labels = copy.copy(cell_labels)
    cyto_labels[nuc_mask] = 0

    nuc_labels = copy.copy(cell_labels)
    nuc_labels[~nuc_mask] = 0

    masks = np.zeros((3, 1024, 1024))
    masks[0, :, :] = cell_labels
    masks[1, :, :] = nuc_labels
    masks[2, :, :] = cyto_labels
    cell_arrays = helper_functions.segment_data_function(image_data, masks)

