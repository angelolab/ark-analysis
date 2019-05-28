import os
import numpy as np
import skimage.io as io


# load TIFs from GUI-based directory structure
# TODO: modularize this so that loading is easier. Create loading helper functions?
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/kausi/20190517_brain_cohort_nucleus_only/'

# get names of each, clean up for subsequent saving
folders = os.listdir(base_dir + 'no_fftnoise')
folders = [folder for folder in folders if 'Point' in folder]
folders.sort()

# load all data into a single numpy array
data = np.zeros((len(folders), 1024, 1024), dtype='float16')
# axes on data: training run, image, x_dim, y_dim, output_mask
for i in range(len(folders)):
    data[i, :, :] = io.imread(os.path.join(base_dir, 'no_fftnoise', folders[i], 'TIFs/HistoneH3Lyo.tif'))

np.save(base_dir + 'Nuclear_Channel_Input', data)

data_deep = np.zeros((51, 1024, 1024, 3), dtype='float32')
data_deep[0:10, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_0.10.npy')
data_deep[10:20, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_10.20.npy')
data_deep[20:30, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_20.30.npy')
data_deep[30:40, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_30.40.npy')
data_deep[40:50, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_40.50.npy')
data_deep[50:51, ...] = np.load(base_dir + 'segmentation_masks/' + 'model_output_50.60.npy')
# save back to same folder structure
# save images back to folder for viewing from regular network
for i in range(len(folders)):
    if data_deep.shape[-1] == 3:
        # three category network
        border_idx = 0
        nuc_idx = 1
        io.imsave(os.path.join(base_dir, 'segmentation_masks', folders[i] + 'nuc_interior.tiff'), data_deep[i, :, :, nuc_idx])
        io.imsave(os.path.join(base_dir, 'segmentation_masks', folders[i] + 'nuc_border.tiff'), data_deep[i, :, :, border_idx])
    else:
        # 4 category network
        border_idx = [0, 1]
        nuc_idx = 2
        io.imsave(os.path.join(image_dir, names[i] + '_nucleus.tiff'), data[i, 3, :, :, nuc_idx])
        io.imsave(os.path.join(image_dir, names[i] + '_border.tiff'),
                  data[i, 3, :, :, border_idx[0]] + data[i, 3, :, :, border_idx[1]])
