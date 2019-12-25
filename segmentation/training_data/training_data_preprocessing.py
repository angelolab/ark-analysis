import skimage.io as io
from segmentation.utils import data_utils
import importlib
import os
import numpy as np
import xarray as xr


# create npz array of labeled images for training
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191216_Mega_DNA/'
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/shirley/test_points/big_test/'
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/shirley/test_points/big_test/caliban_v1/training_data/'
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/selena/20190925_PAH_project/PAHTrainingData/caliban_v3/fixed/'

data_name = "R6_Point5"

# load data from previously created xarray
training_data_x = xr.open_dataarray(base_dir + data_name + "_X.nc")
training_data_y = xr.open_dataarray(base_dir + data_name + "_y.nc")

# or create one now
training_data_x = data_utils.load_tifs_from_points_dir(base_dir, tif_folder="", tifs=["Na.tif"], points=["R6_Point5"])
io.imshow(training_data_x[0, :, :, 0])

training_data_y = data_utils.load_tifs_from_points_dir(base_dir, tif_folder="", tifs=["caliban_annotation.tiff"], points=["R6_Point5"])
io.imshow(training_data_y[0, :, :, 0])

# subset data if only a portion will be used
training_data_x = training_data_x[:, :, :396, :]
training_data_y = training_data_y[:, :, :396, :]

# add blank channels if missing from imaging run
channel_order = ["H3", "NaK ATPase", "Lamin AC"]
non_blank_channels = ["H3", "NaK ATPase"]
training_data_x = data_utils.reorder_xarray_channels(channel_order=channel_order, channel_xr=training_data_x,
                                                           non_blank_channels=non_blank_channels)

training_data_x.to_netcdf(base_dir + data_name + "_X.nc")
training_data_y.to_netcdf(base_dir + data_name + "_y.nc")

# separate out points that will become test points
training_data_x_test = training_data_x.loc[training_data_x.points == "Point12"]
training_data_x = training_data_x.loc[training_data_x.points != "Point12"]
training_data_y_test = training_data_y.loc[training_data_y.points == "Point12"]
training_data_y = training_data_y.loc[training_data_y.points != "Point12"]

np.savez(base_dir + data_name + "_test.npz", X=training_data_x_test, y=training_data_y_test)


# crop data to appropriate size
crop_size = 256
stride = 0.3
training_data_x_cropped = data_utils.crop_image_stack(training_data_x, crop_size=crop_size, stride_fraction=stride)
training_data_y_cropped = data_utils.crop_image_stack(training_data_y, crop_size=crop_size, stride_fraction=stride)

if training_data_y_cropped.shape[:-1] != training_data_x_cropped.shape[:-1]:
    raise ValueError("cropped arrays have different sizes")
else:
    print("looks good")

np.savez(base_dir + data_name + "_{}x{}_stride_{}.npz".format(crop_size, crop_size, stride),
         X=training_data_x_cropped, y=training_data_y_cropped)


# combine different npzs together
npz1 = np.load(base_dir + "R1_Point26_256x256_stride_0.3.npz")
npz2 = np.load(base_dir + "R1_Point32_256x256_stride_0.3.npz")
npz3 = np.load(base_dir + "R6_Point5_256x256_stride_0.3.npz")

combined_x = np.concatenate((npz1["X"], npz2["X"], npz3["X"]), axis=0)
combined_y = np.concatenate((npz1["y"], npz2["y"], npz3["y"]), axis=0)

np.savez(base_dir + "PAH_Caliban_V3_clean.npz", X=combined_x, y=combined_y)
