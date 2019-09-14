# Ilastik segmentation pipeline
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import h5py
import os

# create labeled data for ilastik training. Requires each class to have a distinct pixel value in image
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/'
save_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/ilastik/First_Run/'

points = os.listdir(base_dir)
points = [point for point in points if 'Point' in point]

zips = os.listdir(base_dir + 'Zips')

# TODO refactor into helper function
for point in points:
    # select TIFs corresponding to different label classes
    id = 'Nuc'
    contours = [contour for contour in zips if point in contour and id in contour and '.tif' in contour]
    contours.sort()

    # get border, interior and bg mask
    border = contours[0]
    interior = contours[1]
    border = io.imread(border)
    interior = io.imread(border)
    bg = np.logical_and(border < 1, interior < 1)

    # give each class distinct integer value in single tif
    ilastik_tif = np.zeros((1024, 1024), dtype='uint8')
    ilastik_tif[border > 0] = 1
    ilastik_tif[interior > 0] = 2
    ilastik_tif[bg > 0] = 3

    io.imsave(save_dir + point + 'ilastik_labels.tif', ilastik_tif)


# convert computed ilastic probabilities into TIFs for watershed processing
# TODO: make section in training_freeze_1 Input_Data subfolder that has ilastic

output_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/Ilastik/Point1_12_18_23_3X/Point23/raw/'

output_file = h5py.File(output_dir + '_________________Probabilities.h5', 'r')

list(output_file.keys())

output = output_file['exported_data']

x = output[1, :, :]
io.imshow(x)