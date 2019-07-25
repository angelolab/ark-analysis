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


# load segmentation masks
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
seg_folder = ''
cell_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, ['Point23'], ['Nuclear_Mask_Label.tif'])
nuc_seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, ['Point23'], ['Nuclear_Mask_Label.tif'])

# load TIFs
tif_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
tif_folder = 'TIFsNoNoise'
image_data = helper_functions.load_tifs_from_points_dir(tif_dir, tif_folder, ['Point23'], ['dsDNA.tif', 'LaminAC.tif'])

save_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/segmented_data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for point in image_data.point.values:

    # get segmentation masks
    cell_labels = cell_seg_data.loc[point, 'Nuclear_Mask_Label.tif', :, :].values.astype('int')

    # merge small cells?
    cell_props = skimage.measure.regionprops(cell_labels)

    nuc_mask = nuc_seg_data.loc[point, 'Nuclear_Mask_Label.tif', :, :].values.astype('int')

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

    # TODO data normalization and scaling

    if not os.path.exists(os.path.join(save_dir, point)):
        os.makedirs(os.path.join(save_dir, point))
    cell_data.to_netcdf(os.path.join(save_dir, point, 'segmented_data.nc'))





fake_masks = np.zeros((3, 20, 20))
fake_masks[0, 0:8, 0:8] = 1
fake_masks[1, 3:6, 3:6] = 1
fake_masks[2, :, :] = fake_masks[0, :, :] - fake_masks[1, :, :]

fake_masks[:, 10:17, 10:17] = fake_masks[:, 0:7, 0:7] * 2


fake_data = np.ones((4, 20, 20))
fake_data[2, ...] = fake_data[2, ...] * 3
fake_data[3, ...] = fake_data[3, ...] * 6
fake_data[1, ...] = fake_data[1, ...] * 2


fake_data_xr = xr.DataArray(fake_data, coords=[['Channel_1', 'Channel_2', 'Channel3', 'Channel4'],
                                               range(fake_data.shape[1]), range(fake_data.shape[1])],
                            dims=['channel', 'x_axis', 'y_axis'])

fake_masks_xr = xr.DataArray(fake_masks, coords=[['cell_mask', 'nuc_mask', 'cyto_mask'], range(20), range(20)],
                                      dims=['subcell_loc', 'rows', 'cols'])

x = segment_images(fake_data_xr, fake_masks_xr)

io.imshow(fake_data[0, :, :])