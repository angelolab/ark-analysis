# generates a watershed transform of probability masks

import numpy as np
import skimage.measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io as io
import xarray as xr
import skimage.filters.rank as rank
import skimage.morphology as morph
import copy
import scipy.ndimage as nd


# watershed generation from deepcell transformed data
# read in relavant files
mask_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190903_subsampling/'
deepcell_name = 'Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_27Point8.npy'
fgbg_name = "Training_Freeze_1_Nuc_fgbg_256_dense_64_conv_epoch_06_Point8metrics.npy"

# cell data
deepcell_name = 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_27Point8.npy'
fgbg_name = "Training_Freeze_1_81_rf_fgbg_256_dense_64_conv_epoch_06_Point8metrics.npy"


watershed_name = 'Training_Freeze_1_Nuc_watershed_81_rf_256_dense_64_conv_2erosion_epoch_12point8.npy_watershed'


# HH3 data
deepcell_name = 'Training_Freeze_1_Nuc_HH3_81_rf_512_dense_128_conv_epoch_18_point8metrics.npy'
fgbg_name = "Training_Freeze_1_Nuc_HH3_fgbg_81_rf_256_dense_64_conv_epoch_04_point8metrics.npy"
watershed_name = 'Training_Freeze_1_Nuc_HH3_watershed_81_rf_256_dense_64_conv_epoch_18_point8metrics.npy_watershed'


seeds = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190823_TA489_Redo/zips/Point8_Cell_Mask_Label.tif')
seed_props = skimage.measure.regionprops(seeds)

seed_array = np.zeros((1024, 1024))
for i in range(len(seed_props)):
    coords = np.floor(seed_props[i].centroid).astype('int')
    seed_array[coords[0], coords[1]] = 5

maxs = seed_array > 1



border_probs = io.imread(mask_dir + deepcell_name + '_border.tiff')
interior_probs = io.imread(mask_dir + deepcell_name + '_interior.tiff')
watershed_probs = io.imread(mask_dir + watershed_name + '_smoothed_probs.tiff')
watershed_probs = io.imread(mask_dir + "Training_Freeze_1_Nuc_watershed_81_rf_256_dense_64_conv_2erosion_epoch_12_point8metrics.npy_watershed_smoothed_probs.tiff")
fgbg_probs = io.imread(mask_dir + fgbg_name + '_fgbg.tiff')

maxs = watershed_probs > 2

smoothed_fgbg = nd.gaussian_filter(fgbg_probs, 2)
fgbg_mask = smoothed_fgbg > 0.3

smoothed_interior = nd.gaussian_filter(interior_probs, 2)
interior_mask = smoothed_interior > 0.15


io.imshow(smoothed_interior > 0.6)

# # calculate maxs from smoothed nuclear mask
# maxs = peak_local_max(mask_nuc_smoothed, indices=False, min_distance=5)

# # read in matlab generated mask
# maxs = plt.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_maxs_matlab_0.05prominance_threshold.tiff')
# maxs = maxs[:, :, 0]

# use maxs to generate watershed
markers = skimage.measure.label(maxs, connectivity=1)

# remove any maxs that are 4 pixels or smaller
for cell in np.unique(markers):
    mask = markers == cell
    if np.sum(mask) < 5:
        markers[mask] = 0


# watershed over border mask vs negative interior mask?
labels = np.array(watershed(border_probs, markers, mask=interior_mask, watershed_line=1))

io.imsave(mask_dir + deepcell_name + '_label_mask_0.15_interior_threshold_2erosion.tiff', labels)

# pixel expansion
expanded = morph.dilation(labels, selem=morph.square(7))
io.imsave(mask_dir + deepcell_name + "pixel_expansion_7.tiff", expanded)


# for pipeline
deepcell_masks = xr.open_dataarray(mask_dir + 'deepcell_network_output.nc')
watershed_masks = xr.open_dataarray(mask_dir + 'Decidua_watershed_101_rf_64_conv_256_dense_epoch_03_watershed_network_output.nc')
deep_points = deepcell_masks.coords["points"].values
water_points = watershed_masks.coords["points"].values
border_mask = deepcell_masks.loc[deep_points[point], :, :, 'border_mask']
interior_mask = deepcell_masks.loc[deep_points[point], :, :, 'interior_mask']
