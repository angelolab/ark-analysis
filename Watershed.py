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


seeds = io.imread('/Users/noahgreenwald/Google Drive/Grad School/Lab/Segmentation_Contours/Decidua/Zips/Point12_Objects_Mask_Label.tif')
seed_props = skimage.measure.regionprops(seeds)

seed_array = np.zeros((1024, 1024))
for i in range(len(seed_props)):
    coords = np.floor(seed_props[i].centroid).astype('int')
    seed_array[coords[0], coords[1]] = 5

maxs = seed_array > 1


# watershed generation from deepcell transformed data
# read in relavant files
mask_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190731_decidua_object_test/'

deepcell_masks = xr.open_dataarray(mask_dir + 'deepcell_network_output.nc')
watershed_masks = xr.open_dataarray(mask_dir + 'Decidua_watershed_101_rf_64_conv_256_dense_epoch_03_watershed_network_output.nc')
label_masks = []

for point in range(len(watershed_masks.coords['points'])):
    deep_points = deepcell_masks.coords["points"].values
    water_points = watershed_masks.coords["points"].values
    border_mask = deepcell_masks.loc[deep_points[point], :, :, 'border_mask']
    interior_mask = deepcell_masks.loc[deep_points[point], :, :, 'interior_mask']
    maxs = watershed_masks.loc[water_points[point], :, :, 'smoothed_watershed_probs']
    maxs = maxs > 2
    background_mask = watershed_masks.loc[water_points[point], :, :, 'background_mask']
    cell_mask = background_mask < 0.93
    cell_mask = rank.median(cell_mask, np.ones((5,5)))

    # # calculate maxs from smoothed nuclear mask
    # maxs = peak_local_max(mask_nuc_smoothed, indices=False, min_distance=5)

    # # read in matlab generated mask
    # maxs = plt.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_maxs_matlab_0.05prominance_threshold.tiff')
    # maxs = maxs[:, :, 0]

    # use maxs to generate watershed
    markers = skimage.measure.label(maxs, connectivity=1)

    # watershed over border mask vs negative interior mask?
    labels = watershed(-interior_mask, markers, mask=cell_mask, watershed_line=1)
    label_masks.append(labels)

label_masks = np.array(label_masks)