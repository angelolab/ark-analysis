import os
import numpy as np
import skimage.io as io
import scipy.ndimage as nd


# load TIFs from GUI-based directory structure
base_dir = '/Users/noahgreenwald/Documents/MIBI_DATA/JP/'

# get names of each, clean up for subsequent saving
folders = os.listdir(base_dir + 'NoBgNoNoise')
folders = [folder for folder in folders if 'Point' in folder]
folders.sort()

# load all data into a single numpy array
data = np.zeros((len(folders), 1024, 1024), dtype='float16')
# axes on data: training run, image, x_dim, y_dim, output_mask
for i in range(len(folders)):
    data[i, :, :] = io.imread(os.path.join(base_dir, 'NoBgNoNoise', folders[i], 'TIFs/HH3.tif'))

np.save(base_dir + 'Nuclear_Channel_Input', data)

# run deepcell, save output back to folder
data_deep = np.load(base_dir + '05Jul19_Vestro/' + 'model_output.npy')

# save back to same folder structure
# save images back to folder for viewing from regular network

for i in range(len(folders)):
    if data_deep.shape[-1] == 3:
        # 3-class network
        border_idx = 0
        nuc_idx = 1
        smoothed = nd.gaussian_filter(data_deep[i, :, :, nuc_idx], 5)
        io.imsave(os.path.join(base_dir, 'segmentation_masks', folders[i] + 'nuc_interior.tiff'), data_deep[i, :, :, nuc_idx])
        io.imsave(os.path.join(base_dir, 'segmentation_masks', folders[i] + 'nuc_border.tiff'), data_deep[i, :, :, border_idx])
        io.imsave(os.path.join(base_dir, 'segmentation_masks', folders[i] + 'nuc_interior_smoothed.tiff'), smoothed)


