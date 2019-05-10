# generates a watershed transform of probability masks

import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io as io
import copy

# read in relavant files
image_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/cnn_data/Deepcell_docker/output/190430_watershed_test/'
mask_nuc = io.imread(image_dir + 'interior_border_30_nucleus.tiff')
mask_border = io.imread(image_dir + 'interior_border_30_border.tiff')
mask_truth = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/Point23/Nuclear_Interior_Mask_Label.tif')

# create array to hold thresholded probabilities at different values to determine optimal cutoff
mask = np.zeros((1024,1024,5))
temp_mask = np.zeros((1024,1024,1))
temp_mask[:, :, 0] = data[2, 3, :, :, 1]
mask[:, :, :] = copy.copy(temp_mask)

io.imshow(mask[:, :, 1])
mask[mask[:, :, 0] < 0.8, 1] = 0
mask[np.logical_or(mask[:, :, 0] < 0.7, mask[:, :, 0] > 0.8), 2] = 0
mask[np.logical_or(mask[:, :, 0] < 0.6, mask[:, :, 0] > 0.7), 3] = 0
mask[mask[:, :, 0] < 0.7, 4] = 0

# plot thresholded probabilities
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(mask[:, :, 4])
ax[1].imshow(mask_truth)
fig.tight_layout()

# identify maxs for watershed
maxs = peak_local_max(mask[:, :, 4], indices=False, min_distance=3)
markers = skimage.measure.label(mask[:, :, 4] > 0, connectivity=1)
labels = watershed(-mask_nuc, markers, mask=mask_nuc > 0.15, watershed_line=1)
io.imsave(image_dir + 'mask_python_watershed_15.tiff', labels)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(mask_nuc, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Nuclear Probs')
ax[1].imshow(mask_nuc_thresh, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(maxs, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title('Separated objects')


# watershed network transform

watershed_images = []
fg_thresh = io.imread('path_to_threshold')
smoothed_probs = io.imread('path_to_smoothed_probs')

image = fg_thresh
distance = smoothed_probs
local_maxi = peak_local_max(smoothed_probs, min_distance=15, exclude_border=False, indices=False, labels=image)
local_maxi_easy = smoothed_probs > 2
local_maxi_mat = io.imread('matlab_local_maxima')
markers = label(local_maxi)
segments = watershed(-distance, markers, mask=image, watershed_line=True)
watershed_images.append(segments)

watershed_images = np.array(watershed_images)
watershed_images = np.expand_dims(watershed_images, axis=-1)