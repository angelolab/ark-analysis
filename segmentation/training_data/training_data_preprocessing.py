import skimage.io as io
from segmentation import helper_functions
import importlib
import os
import numpy as np


# create overlapping crops for fully convolutational model
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191112_lab_combined/'
training_data_x = helper_functions.load_tifs_from_points_dir(base_dir + "lab_combined_test",
                                                           points=os.listdir(base_dir + "lab_combined_test")[1:],
                                                           tif_folder="raw",
                                                             tifs=["Nuclear_Interior.tif", "Cell_Border.tif", "Nuclear_Border.tif"])

training_data_y = helper_functions.load_tifs_from_points_dir(base_dir + "lab_combined_test",
                                                           points=os.listdir(base_dir + "lab_combined_test")[1:],
                                                           tif_folder="annotated")

training_data_x_test = training_data_x.loc[training_data_x.points == "Point1"]
training_data_x = training_data_x.loc[training_data_x.points != "Point1"]

training_data_y_test = training_data_y.loc[training_data_y.points == "Point1"]
training_data_y = training_data_y.loc[training_data_y.points != "Point1"]

training_data_x_cropped = helper_functions.crop_image_stack(training_data_x, crop_size=256, stride_fraction=0.5)
training_data_y_cropped = helper_functions.crop_image_stack(training_data_y, crop_size=256, stride_fraction=0.5)

training_data_x_test_cropped = helper_functions.crop_image_stack(training_data_x_test, crop_size=256, stride_fraction=1)
training_data_y_test_cropped = helper_functions.crop_image_stack(training_data_y_test, crop_size=256, stride_fraction=1)

np.savez(base_dir + "lab_combined_train/lab_combined_train_256x256.npz", X=training_data_x_cropped, y=training_data_y_cropped)
