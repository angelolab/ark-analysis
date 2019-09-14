import os
import numpy as np
import skimage.io as io
import xarray as xr
import helper_functions


# load TIFs from GUI-based directory structure
base_dir = '/Users/noahgreenwald/Documents/MIBI_DATA/tyler/190808_DCISCOHORT/'
data_folder = ''

# get names of each, clean up for subsequent saving
folders = os.listdir(base_dir + data_folder)
folders = [folder for folder in folders if 'Point' in folder]

# load all data into a single numpy array
data = np.zeros((len(folders), 1, 1024, 1024), dtype='float16')

# axes on data: training run, image, x_dim, y_dim, output_mask
for i in range(len(folders)):
    data[i, 0, :, :] = io.imread(os.path.join(base_dir, data_folder, folders[i], 'TIFs/HH3.tif'))

data_xr = xr.DataArray(data, coords=[folders, ['HH3'], range(1024), range(1024)],
                       dims=['point', 'channels', 'rows', 'cols'])

data_xr.to_netcdf(base_dir + 'Nuclear_Channel_Input.nc')
np.save(base_dir + 'Nuclear_Channel_Input', data)

# run deepcell, save output back to folder, then load here
data_deep = np.load(base_dir + 'model_output.npy')
data_xr = xr.open_dataarray(base_dir + 'Nuclear_Channel_Input.nc')
folders = data_xr.coords['point'].values.tolist()

# TODO: modify deepcell so that it can read in xarray data
folders = np.load(base_dir + 'folder_names.npy')
folders = [x for x in folders]
# extract individual TIFs and save segmentation results back into matlab-compatible folder structure
helper_functions.save_deepcell_tifs(data_deep, folders, base_dir + '/segmentation_output', cohort=True, watershed=False)
