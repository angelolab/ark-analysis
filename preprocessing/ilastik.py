# Ilastik segmentation pipeline
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import h5py
import os

# create labeled data for ilastik training. Requires each class to have a distinct pixel value in image
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/'
save_dir = base_dir + 'Input_Data/Ilastik_Training_Freeze_1/'
os.makedirs(save_dir)

points = os.listdir(base_dir + 'Input_Data/Training_Freeze_1')
points = [point for point in points if 'Point' in point]

zips = os.listdir(base_dir + 'Zips')

# TODO refactor into helper function and modify pixel transformation code for when we directly generate label map
for point in points:
    # select TIFs corresponding to different label classes
    id = 'Cell'
    contours = [contour for contour in zips if point in contour and id in contour and '.tif' in contour]
    contours.sort()

    # get border, interior and bg mask
    border = contours[0]
    interior = contours[1]
    border = io.imread(base_dir + 'Zips/' + border)
    interior = io.imread(base_dir + 'Zips/' + interior)
    bg = np.logical_and(border < 1, interior < 1)

    # give each class distinct integer value in single tif
    ilastik_tif = np.zeros((1024, 1024), dtype='uint8')
    ilastik_tif[border > 0] = 1
    ilastik_tif[interior > 0] = 2
    ilastik_tif[bg > 0] = 3

    io.imsave(save_dir + point + id + '_ilastik_labels.tif', ilastik_tif)


# convert computed ilastic probabilities into TIFs for watershed processing
output_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/Ilastik/Point1_12_18_23_3X/Point23/raw/'

output_file = h5py.File(output_dir + '_________________Probabilities.h5', 'r')

list(output_file.keys())

output = output_file['exported_data']

x = output[1, :, :]
io.imshow(x)