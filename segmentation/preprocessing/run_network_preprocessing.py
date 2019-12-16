import os
import numpy as np
import skimage.io as io
import xarray as xr
from segmentation import helper_functions
import importlib
importlib.reload(helper_functions)


# load TIFs from GUI-based directory structure
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/selena/20190925_PAH_project/PAHTrainingData/no_noise/'
base_dir = "/Users/noahgreenwald/Documents/MIBI_Data/shirley/test_points/big_test/denoised/"
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191112_lab_combined/lab_combined_train"
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/selena/20191215_GBM_Cohort/TA_551_test/no_noise/'

# optionally, specify a set of channels to be summed for analysis
test_points = os.listdir(base_dir)
test_points = ['10_31739_6_13', '10_31739_6_14', '10_31740_6_11', '10_31740_6_12', '10_31743_16_7', '10_31743_16_8',
               '12_31750_16_12', '12_31750_1_11', '12_31754_16_15', '12_31754_18_2', '14_31755_18_1', '14_31755_2_1',
               '14_31756_16_16', '14_31756_2_2', '14_31758_2_4', '14_31758_2_5', '16_31762_16_21', '16_31762_2_9',
               '16_31765_18_3', '16_31765_19_1', '16_31768_17_2', '16_31768_18_7', '16_31770_4_1', '16_31770_4_2',
               '18_31776_14_12', '18_31778_5_3', '18_31779_5_4', '18_31783_14_10', '18_31783_14_11', '18_31783_14_9',
               '6_31725_8_2', '6_31725_8_3', '6_31726_8_4', '6_31726_8_5', '6_31727_8_10', '6_31727_8_9',
               '8_31734_11_18', '8_31734_12_1', '8_31735_12_6', '8_31735_12_7', '8_31736_12_15', '8_31736_12_16']

test_points = ["Point8_TA489_run", "Point1_first_run"]
test_points = ["Point12_first_run"]

sum_channels_xr = helper_functions.load_tifs_from_points_dir(base_dir, tif_folder="TIFs", points=test_points,
                                                             tifs=["CD45.tif", "CD3.tif", "CD4.tif", "CD8.tif",
                                                                   "CD14.tif", "CD206.tif", "CD163.tif", "DCSIGN.tif",
                                                                   "CD56.tif", "HLADR.tif"])

sum_channels_xr = helper_functions.load_tifs_from_points_dir(base_dir, tif_folder="TIFs", points=test_points,
                                                             tifs=["CD3.tif", "CD206.tif", "CD163.tif", "CD56.tif"])


channel_sum = np.sum(sum_channels_xr.values, axis=3, dtype="uint8")

summed_xr = xr.DataArray(np.expand_dims(channel_sum, axis=-1), coords=[sum_channels_xr.points, sum_channels_xr.rows,
                                                                       sum_channels_xr.cols, ["summed_channel"]],
                         dims=sum_channels_xr.dims)

tif_saves = ["summed_channel"]
for point in summed_xr.points.values:
    os.makedirs(os.path.join(base_dir, "channel_sums", point))
    for tif in tif_saves:
        save_path = os.path.join(base_dir, "channel_sums", point, tif + ".tif")
        io.imsave(save_path, summed_xr.loc[point, :, :, tif].values.astype('uint8'))

points = os.listdir(base_dir)
points.sort()
points = points[:-2]
# load channels to be included in deepcell data
data_xr = helper_functions.load_tifs_from_points_dir(base_dir, tif_folder='TIFs',
                                                    tifs=['HH3.tif'], dtype='int8')
                                                     #tifs=['H3.tif', "VIM.tif", "HLAG.tif", "CD14.tif", "CD56.tif", "CD3.tif"])


# add blank TIFs to channels not present
# set current tif order so that channel types match up
channel_order = ["H3.tif", "VIM.tif", "HLAG.tif", "CD3", "CD14.tif", "CD56.tif"]
channel_order = ["HH3", "NaKATPase", "LaminAC"]

non_blank_channels = ["HH3"]

data_xr = helper_functions.reorder_xarray_channels(channel_order=channel_order, channel_xr=data_xr,
                                                   non_blank_channels=non_blank_channels)

# save xarray
data_xr.to_netcdf(base_dir + 'Deepcell_Input.nc', format="NETCDF3_64BIT")


# subset the array for saving if too large
np.savez(base_dir + "Deepcell_Input_Meyeloid.npz", X=data_xr.values)


data_xr = data_xr[:, :2, :2, :]
data_xr.to_netcdf(base_dir + 'Deepcell_Input_myeloid_labels.nc', format="NETCDF3_64BIT")
