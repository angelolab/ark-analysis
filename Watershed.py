# generates a watershed transform of probability masks

import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
import skimage.io as io
import copy

# watershed generation from deepcell transformed data
# read in relavant files
image_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190731_decidua_object_test/'
mask_nuc_smoothed = io.imread(image_dir + '3_class_64_filters_256_densefilters_epoch_30_nucleus_smoothed_5.tiff')
mask_nuc = io.imread(image_dir + '3_class_64_filters_256_densefilters_epoch_30_nucleus.tiff')
mask_border = io.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_border.tiff')

# identify maxs for watershed
# calculate maxs from smoothed nuclear mask
maxs = peak_local_max(mask_nuc_smoothed, indices=False, min_distance=5)

# read in matlab generated mask
maxs = plt.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_maxs_matlab_0.05prominance_threshold.tiff')
maxs = maxs[:, :, 0]

# read in watershed network generated maxs
maxs = io.imread(image_dir + 'watershed_64_filters_400_densefilters_balanced_epoch_20_smoothed_probs.tiff')
maxs = maxs > 2

# use maxs to generate watershed
markers = skimage.measure.label(maxs, connectivity=1)
labels = watershed(-mask_nuc, markers, mask=mask_nuc > 0.15, watershed_line=1)
io.imshow(labels)
io.imsave(image_dir + '3_class_64_filters_256_densefilters_epoch_30_mask_python_max_watershed.tiff', labels)




# watershed network transform
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/analyses/'
watershed_images = []
fg_thresh = io.imread(base_dir + '20190505_watershed_retrain/' + 'watershed_epoch_30_nucleus.tiff')
smoothed_probs = io.imread(base_dir + '20190621_postprocessing/' + '3_class_w_interior_and_watershed_watershed_epoch_20_smoothed_probs.tiff')
prob_map = io.imread(base_dir + '20190621_postprocessing/' + '3_class_w_interior_and_watershed_epoch_20_nucleus.tiff')

image = prob_map > 0.2
distance = prob_map
local_maxi_easy = smoothed_probs > 2
markers = label(local_maxi_easy)
segments = watershed(-distance, markers, mask=image, watershed_line=True)
watershed_images.append(segments)

watershed_images = np.array(watershed_images)
watershed_images = np.expand_dims(watershed_images, axis=-1)

io.imshow(watershed_images[0, :, :, 0])

io.imsave(base_dir + '20190621_postprocessing/' + 'mask_3_class_w_interior_and_watershed_watershed_epoch_20.tiff', watershed_images[0, :, :, 0])