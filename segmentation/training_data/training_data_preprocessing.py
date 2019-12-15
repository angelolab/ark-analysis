import skimage.io as io
from segmentation import helper_functions
import importlib
import os
import numpy as np
import xarray as xr


# create npz array of labeled images for training
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20190615_Decidua/Input_Data/20191212_Decidua_Whole_Cell/'
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/shirley/test_points/big_test/'
base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/shirley/test_points/big_test/caliban_v1/training_data/'

data_name = "16_31768_17_2"

# load data from previously created xarray
training_data_x = xr.open_dataarray(base_dir + data_name + "_X.nc")
training_data_y = xr.open_dataarray(base_dir + data_name + "_y.nc")

# or create one now
training_data_x = helper_functions.load_tifs_from_points_dir(base_dir + "denoised",
                                                           points=["6_31725_8_2"],
                                                           tif_folder="TIFs",
                                                             tifs=["H3.tif", "VIM.tif", "HLAG.tif", "CD3.tif",
                                                                   "CD14.tif", "CD56.tif"], dtype="int8")
training_data_y = helper_functions.load_tifs_from_points_dir(base_dir + "caliban_v1/training_data/",
                                                             points=["16_31768_17_2"],
                                                             tif_folder="", dtype="int32")


# subset data if only a portion will be used
training_data_x = training_data_x[:, 1024:, 1024:, :]
training_data_y = training_data_y[:, 1024:, 1024:, :]

# add blank channels if missing from imaging run
channel_order = ["H3", "NaK ATPase", "Lamin AC"]
non_blank_channels = ["H3", "NaK ATPase"]
training_data_x = helper_functions.reorder_xarray_channels(channel_order=channel_order, channel_xr=training_data_x,
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
training_data_x_cropped = helper_functions.crop_image_stack(training_data_x, crop_size=crop_size, stride_fraction=0.5)
training_data_y_cropped = helper_functions.crop_image_stack(training_data_y, crop_size=crop_size, stride_fraction=0.5)


np.savez(base_dir + data_name + "_{}x{}.npz".format(crop_size, crop_size),
         X=training_data_x_cropped, y=training_data_y_cropped)


# combine different npzs together

base_npz = np.load("/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20190615_Decidua/Input_Data/20191212_Decidua_Whole_Cell/20191212_Decidua_Whole_Cell_Train_256x256.npz")

npz1 = np.load(base_dir + "6_31725_8_2.npz")
npz2 = np.load(base_dir + "16_31768_17_2.npz")
npz3 = np.load(base_dir + "CD3_Segmentation_Supplement.npz")
npz4 = np.load(base_dir + "Nuclear_Segmentation_Supplement.npz")

combined_x = np.concatenate((base_npz["X"], npz2["X"], npz3["X"], npz4["X"]), axis=0)
combined_y = np.concatenate((base_npz["y"], npz2["y"], npz3["y"], npz4["y"]), axis=0)

np.savez(base_dir + "20191214_Decidua_Caliban_V1_no6week.npz", X=combined_x, y=combined_y)
