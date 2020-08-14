# takes data from ilastik/selena pipeline and conver to xarray
import numpy as np
import h5py
import xarray as xr
import skimage.io as io
import os

base_dir = "/Users/noahgreenwald/Documents/MIBI_Data/selena/20191215_GBM_Cohort/"

h5_files = os.listdir(base_dir)
h5_files = [file for file in h5_files if "Prob_" in file]

xr_input = np.zeros((len(h5_files), 1024, 1024, 1), dtype="int16")

for idx in range(len(h5_files)):
    h5 = h5py.File(base_dir + h5_files[idx], 'r')
    h5_stack = h5["exported_data"]

    # only 100 values stored in each slice, convert to unique integers
    input_img = np.zeros((1024, 1024), dtype='int16')
    for i in range(h5_stack.shape[-1]):
        im_slice = h5_stack[:, :, i] * 100 * (i + 1)
        input_img[:, :] = input_img[:, :] + im_slice

    xr_input[idx, :, :, 0] = input_img


