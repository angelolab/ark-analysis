import numpy as numpy
import os
import sys

sys.path.append(os.path.abspath('../deepcell-tf'))
from deepcell import utils

# scripts to load training data and process for caliban
example_npz = np.load('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/caliban/desktop/stack_041_test_all_channels.npz')


# get TIFs associated with given point
training_data_path = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/Input_Data/Training_Freeze_1_Nuc/Point5'
labels_data_path = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/Input_Data/Training_Freeze_1_Nuc/Point5'

files = os.listdir(training_data_path + '/annotated')
raw = np.zeros((1, 1024, 1024, 5), dtype='int16')

raw_files = os.listdir(folder_path + '/raw')
for i in range(len(raw_files)):
    raw[0, :, :, i] = io.imread(folder_path + '/raw/' + raw_files[i])

# read in labels
annotated = np.zeros((1, 1024, 1024, 1), dtype='int16')
mask = io.imread(labels_data_path + '/annotated/Nuc_Mask_Label.tif')
annotated[0, :, :, 0] = mask

# subsample to smaller resolution
raw_input=raw[:, :200, :200, :]
annotated_input = annotated[:, :200, :200, :]

# bug requires label 1 to be present
annotated_input[:, 4, 4, 0] = 1
np.savez('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/caliban/desktop/mibi_stack.npz',
         raw=raw_input,
         annotated=annotated_input)

