import os
import numpy as np
import skimage.io as io
import xarray as xr
from segmentation import helper_functions

base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/felix/20191001_cohort/'

# load tifs to be summed together

membrane_xr = helper_functions.load_tifs_from_points_dir(base_dir + "no_noise", tif_folder="TIFs",
                                                         tifs=["145_CD45.tif", "166_GLUT1.tif", "174_CK.tif"])

membrane_sum = np.sum(membrane_xr.values, axis=3, dtype="int16")

# load TIFs from GUI-based directory structure

data_xr = helper_functions.load_tifs_from_points_dir(base_dir + "no_noise", tif_folder='TIFs',
                                                     tifs=['089_H3.tif', '145_CD45.tif', '113_vimentin.tif'])
data_xr.name = '20191001_cohort_combined_membrane'

data_xr.values[:, :, :, 1] = membrane_sum

# add blank TIFs to channels not present
# this is the order the network expects: ["BetaTubulin.tif", "H3.tif", "LaminAC.tif", "NaK ATPase.tif", "Vimentin.tif"]
# set current tif order so that channel types match up
tif_order = ["BetaTubulin.tif", "089_H3.tif", "LaminAC.tif", "145_CD45.tif", "113_vimentin.tif"]

blank_tifs = ["BetaTubulin.tif", "LaminAC.tif"]
blank = io.imread("/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/Blank_Tif.tif")

# create temporary array to hold the blank tifs
temp_array = np.zeros((data_xr.shape[:3] + (5,)), dtype='int16')

for i in range(len(tif_order)):
    if tif_order[i] in blank_tifs:
        temp_array[:, :, :, i] = blank
    else:
        temp_array[:, :, :, i] = data_xr.loc[:, :, :, tif_order[i]].values

data_xr_blanked = xr.DataArray(temp_array, coords=[data_xr.points, range(1024), range(1024), range(5)],
                               dims=["points", "rows", "cols", "channels"])

data_xr_blanked.to_netcdf(base_dir + 'Deepcell_Input_membrane_sum.nc', format="NETCDF3_64BIT")

# run deepcell, save output back to folder, then load here
data_xr_watershed = xr.open_dataarray(base_dir + 'watershed_output.nc')
data_xr_pixel = xr.open_dataarray(base_dir + 'deepcell_output_membrane_sum.nc')

data_xr_pixel.name = "Membrane_Sum"
os.makedirs(base_dir + '/segmentation_output_membrane_1smooth')
# extract individual TIFs and save segmentation results back into matlab-compatible folder structure
helper_functions.save_deepcell_tifs(data_xr_pixel, save_path=base_dir + '/segmentation_output_membrane_1smooth',
                                    transform='pixel', pixel_smooth=1)


