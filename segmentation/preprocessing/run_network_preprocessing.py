import os
import numpy as np
import skimage.io as io
import xarray as xr
import helper_functions


# load TIFs from GUI-based directory structure
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/selena/20190925_PAH project/HH3NoNoise_RawMarkersIntensities/TMA1_run1_no_noise/'

data_xr = helper_functions.load_tifs_from_points_dir(base_dir, tif_folder='TIFs', tifs=['H3.tif'])
data_xr.name = 'TMA1_run1_no_noise'
data_xr.to_netcdf(base_dir + 'Nuclear_Channel_Input.nc', format="NETCDF3_64BIT")

# run deepcell, save output back to folder, then load here
data_xr_watershed = xr.open_dataarray(base_dir + 'watershed_output.nc')
data_xr_pixel = xr.open_dataarray(base_dir + 'TMA1_run1_no_noise.nc')
folders = data_xr.coords['points'].values.tolist()

folders = [x for x in folders]
# extract individual TIFs and save segmentation results back into matlab-compatible folder structure
helper_functions.save_deepcell_tifs(data_xr_pixel, save_path=base_dir + '/segmentation_output', transform='pixel')


