# preprocess label masks from hand contoured cells to generate training data

import skimage.io as io
import helper_functions
import importlib
import os
importlib.reload(helper_functions)

# base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/Point23/'
base_dir = '/Users/noahgreenwald/Google Drive/Grad School/Lab/Segmentation_Contours/Decidua/Zips/'

files = os.listdir(base_dir)
files.sort()
total_cell = [file for file in files if 'Objects_Border_Interior' in file]
interior_cell = [file for file in files if 'Objects_Interior' in file]

for i in range(len(total_cell)):
    total_cell_img = io.imread(os.path.join(base_dir, total_cell[i]))
    interior_cell_img = io.imread(os.path.join(base_dir, interior_cell[i]))

    mask = helper_functions.process_training_data(interior_cell_img, total_cell_img)
    base_name = total_cell[i].replace('_Border_Interior_Mask.tif', '')
    io.imsave(os.path.join(base_dir, base_name + '_Mask_Label.tif'), mask)
