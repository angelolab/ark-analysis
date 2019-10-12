import napari
from skimage import data
import skimage.io as io
import numpy as np

labels = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20190606_params/figs/mask_3_class_64_filters_256_densefilters_epoch_30_ground_truth_raw.tiff')
DNA = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/First_Run/Point23/TIFsNoNoise/dsDNA.tif')
NaK = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/First_Run/Point23/TIFsNoNoise/NaK ATPase.tif')
labels = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20190621_postprocessing/Figs/mask__0threshold_0.1cutoff_ground_truth_raw.tiff')

for idx, val in enumerate(np.unique(labels)):
    labels[labels == val] = idx

labels[labels == 328] = 0
randoms = np.random.randint(0, 20, 1024*1024*5)
channels_first = randoms.reshape((5, 1024, 1024))
channels_last = randoms.reshape((1024, 1024, 5))

# example code for trying out napari functionality
with napari.gui_qt():
    viewer = napari.view_image(channels_first, rgb=False, name='channels_first')
    #viewer.add_image(NaK, rgb=False, name='NaK')
    #viewer.add_labels(labels, name='ground truth')


combined[0, :, :] = DNA
combined[1, :, :] = NaK

combined = np.zeros((5, 1024, 1024))
combined[1, :, :] = DNA
combined[0, :, :] = NaK


# example code for trying out napari functionality
with napari.gui_qt():
    viewer = napari.view_image(combined, rgb=False, name='combined')
    viewer.add_image(NaK, rgb=False, name='NaK')
    viewer.add_labels(labels, name='ground truth')


