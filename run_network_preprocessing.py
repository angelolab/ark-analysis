import os
import numpy as np
import skimage.io as io
import xarray as xr
import scipy.ndimage as nd


# load TIFs from GUI-based directory structure
base_dir = '/Users/noahgreenwald/Documents/MIBI_DATA/JP/'

# get names of each, clean up for subsequent saving
folders = os.listdir(base_dir + 'NoBgNoNoise')
folders = [folder for folder in folders if 'Point' in folder]

# load all data into a single numpy array
data = np.zeros((len(folders), 1, 1024, 1024), dtype='float16')
# axes on data: training run, image, x_dim, y_dim, output_mask
# TODO: change to xarray so that we keep folder names
for i in range(len(folders)):
    data[i, 0, :, :] = io.imread(os.path.join(base_dir, 'NoBgNoNoise', folders[i], 'TIFs/HH3.tif'))

data_xr = xr.DataArray(data, coords=[folders, ['HH3'], range(1024), range(1024)],
                       dims=['point', 'channels', 'rows', 'cols'])

np.save(base_dir + 'Nuclear_Channel_Input', data)

# run deepcell, save output back to folder
data_deep = np.load(base_dir + '05Jul19_Vestro/' + 'model_output.npy')

# save back to same folder structure
helper_functions.save_deepcell_tifs(data_deep, folders, base_dir + '/segmentation_output', cohort=True, watershed=False)


