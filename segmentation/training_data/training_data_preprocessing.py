import skimage.io as io
import helper_functions
import importlib
import os
import numpy as np
importlib.reload(helper_functions)

# preprocess label masks from hand contoured cells to generate training data

# base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/Point23/'
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190823_TA489_Redo/zips/'

files = os.listdir(base_dir)
files = [file for file in files if 'Point5' in file]
files.sort()
total_cell = [file for file in files if 'Nuc_Border_Interior' in file]
interior_cell = [file for file in files if 'Nuc_Interior' in file]

for i in range(len(total_cell)):
    total_cell_img = io.imread(os.path.join(base_dir, total_cell[i]))
    interior_cell_img = io.imread(os.path.join(base_dir, interior_cell[i]))

    mask = helper_functions.process_training_data(interior_cell_img, total_cell_img)
    base_name = total_cell[i].replace('_Border_Interior_Mask.tif', '')
    io.imsave(os.path.join(base_dir, base_name + '_Mask_Label.tif'), mask)


# for greycell network, which takes linear combination of markers
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/'
points = os.listdir(base_dir)
points = [point for point in points if 'Point' in point]

for point in points:
    files = os.listdir(os.path.join(base_dir, point))
    files = [file for file in files if 'Point' not in file and '.tif' in file]
    tif_array = np.zeros((1024, 1024, len(files)), dtype = 'int16')
    for i in range(len(files)):
        tif_array[:, :, i] = io.imread(os.path.join(base_dir, point, files[i]))

        summed_tif = np.sum(tif_array, axis=2, dtype='int16')
        io.imsave(os.path.join(base_dir, point, 'Summed_Channels.tif'), summed_tif)