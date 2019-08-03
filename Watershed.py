# generates a watershed transform of probability masks

import numpy as np
import skimage.measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io as io
import xarray as xr
import copy

# watershed generation from deepcell transformed data
# read in relavant files
mask_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190731_decidua_object_test/'

deepcell_masks = xr.open_dataarray(mask_dir + 'deepcell_network_output.nc')
watershed_masks = xr.open_dataarray(mask_dir + 'watershed_network_output.nc')
label_masks = []

for point in deepcell_masks.coords['points']:
    border_mask = deepcell_masks.loc[point, :, :, 'border_mask']
    maxs = watershed_masks.loc[point, :, :, 'smoothed_watershed_probs']
    background_mask = watershed_masks.loc[point, :, :, 'background_mask']
    cell_mask = background_mask < 0.95

    # # calculate maxs from smoothed nuclear mask
    # maxs = peak_local_max(mask_nuc_smoothed, indices=False, min_distance=5)

    # # read in matlab generated mask
    # maxs = plt.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_maxs_matlab_0.05prominance_threshold.tiff')
    # maxs = maxs[:, :, 0]

    # use maxs to generate watershed
    markers = skimage.measure.label(maxs, connectivity=1)

    # watershed over border mask vs negative interior mask?
    labels = watershed(border_mask, markers, mask=cell_mask, watershed_line=1)
    io.imshow(labels)
