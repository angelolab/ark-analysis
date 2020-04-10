import numpy as np
import skimage.io as io
import os
import copy

import pandas as pd
import xarray as xr
import skimage

import importlib
from segmentation.utils import plot_utils, evaluation_utils

importlib.reload(evaluation_utils)


# code to evaluate accuracy of different segmentation contours

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/' \
           'Segmentation_Project/analyses/20200327_Metrics_Comparison/'

y_true = [io.imread(base_dir + "true_labels/true_data.tiff")]
y_pred = []

pred_folders = os.listdir(base_dir)
pred_folders = [folder for folder in pred_folders if 'prediction' in folder]

for folder in pred_folders:
    y_pred_temp = xr.open_dataarray(os.path.join(base_dir, folder, 'segmentation_labels.xr'))
    y_pred.append(y_pred_temp.values[0, :, :, 0])

y_true = y_true * len(y_pred)

# mean average precision
m_ap_array = evaluation_utils.compare_mAP({'y_true': y_true, 'y_pred': y_pred}, np.arange(0.5, 1, 0.1))
plot_utils.plot_mAPs(m_ap_array, np.arange(0.5, 1, 0.1), pred_folders)

