import numpy as numpy
import os
xx = np.load('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/caliban/desktop/stack_041_test_all_channels.npz')

xx.files

annotated = xx['annotated']

raw = xx['raw']

annotated = annotated[3:4, :, :, :1]


folder_path = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190813_combined_data/Input_Data/Training_Freeze_1_Nuc/Point5'

files = os.listdir(folder_path + '/annotated')

annotated = np.zeros((1, 1024, 1024, 1), dtype='int16')
mask = io.imread(folder_path + '/annotated/Nuc_Mask_Label.tif')
mask[mask == 5] = 1
mask[mask > ] = 0


annotated[0, :, :, 0] = mask
annotated[1, :, :, 1] = io.imread(folder_path + '/annotated/Nuc_Mask_Label.tif')
annotated[1, :, :, 0] = io.imread(folder_path + '/annotated/Nuc_Mask_Label.tif')

raw = np.zeros((1, 1024, 1024, 5), dtype='int16')

raw_files = os.listdir(folder_path + '/raw')
for i in range(len(raw_files)):
    raw[0, :, :, i] = io.imread(folder_path + '/raw/' + raw_files[i])


raw_input=raw[:, :200, :200, :]
annotated_input = annotated[:, :200, :200, :]

annotated_input[:, 4, 4, 0] = 1
np.savez('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/caliban/desktop/mibi_stack.npz',
         raw=raw_input,
         annotated=annotated_input)

