# take segmentation masks and TIFs and generate single cell data
import skimage.io as io
import skimage.morphology as morph
import skimage
import helper_functions
import importlib
import copy
import numpy as np
import xarray as xr
import os
importlib.reload(helper_functions)

# TODO: rework for segmentation testing only vs pipeline?
# load segmentation masks
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190822_training_freeze_1/'
seg_folder = ''
cell_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, [''], ['Training_Freeze_1_81_rf_512_dense_128_conv_epoch_42_label_mask.tiff'])
nuc_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, [''], ['Training_Freeze_1_Nuc_fgbg_256_dense_64_conv_epoch_07_fgbg.tiff'])

# load TIFs
tif_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/'
tif_folder = ''
image_data = helper_functions.load_tifs_from_points_dir(tif_dir, tif_folder, ['Point1'], ['HH3.tif', 'LaminAC.tif', 'BetaTubulin.tif'])

save_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190822_training_freeze_1/segmented_data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for point in image_data.point.values:

    # get segmentation masks
    cell_labels = cell_seg_data.loc[:, 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_42_label_mask.tiff', :, :].values.astype('int')

    # merge small cells?
    cell_props = skimage.measure.regionprops(cell_labels)

    nuc_mask = nuc_seg_data.loc[:, 'Training_Freeze_1_Nuc_fgbg_256_dense_64_conv_epoch_07_fgbg.tiff', :, :].values.astype('int')

    # binarize to get nuclear vs non nuclear regions
    nuc_mask = nuc_mask > 0

    # duplicate whole cell data, then subtract nucleus for cytoplasm
    cyto_labels = copy.copy(cell_labels)
    cyto_labels[nuc_mask] = 0

    # nuclear data
    nuc_labels = copy.copy(cell_labels)
    nuc_labels[~nuc_mask] = 0

    # save different masks to single object
    masks = np.zeros((3, 1024, 1024))
    masks[0, :, :] = cell_labels
    masks[1, :, :] = nuc_labels
    masks[2, :, :] = cyto_labels
    segmentation_masks = xr.DataArray(masks, coords=[['cell_mask', 'nuc_mask', 'cyto_mask'], range(1024), range(1024)],
                                      dims=['subcell_loc', 'rows', 'cols'])

    # segment images based on supplied masks
    cell_data = helper_functions.segment_images(image_data.loc[point, :, :, :], segmentation_masks)

    # create version of data normalized by cell size
    cell_data_norm = copy.copy(cell_data)
    cell_size = cell_data.values[:, 1:, 0:1]

    # exclude first row (background cell) and first column (cell size) from area normalization
    cell_data_norm.values[:, 1:, 1:] = np.divide(cell_data_norm.values[:, 1:, 1:], cell_size, where=cell_size > 0)

    # arcsinh transformation
    cell_data_norm_trans = copy.copy(cell_data_norm)
    cell_data_norm_trans.values[:, 1:, 1:] = np.arcsinh(cell_data_norm_trans[:, 1:, 1:])

    if not os.path.exists(os.path.join(save_dir, point)):
        os.makedirs(os.path.join(save_dir, point))
    cell_data.to_netcdf(os.path.join(save_dir, point, 'segmented_data.nc'))
    cell_data_norm.to_netcdf(os.path.join(save_dir, point, 'segmented_data_normalized.nc'))
    cell_data_norm_trans.to_netcdf(os.path.join(save_dir, point, 'segmented_data_normalized_transformed.nc'))

