import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import importlib
import helper_functions
import xarray as xr
importlib.reload(helper_functions)


# get directory where images are located
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses'
image_dir = base_dir + '/20190917_naming/'
plot_dir = image_dir + '/figs/'

# get names of each, clean up for subsequent saving
files = os.listdir(image_dir)
files = [file for file in files if 'output.nc' in file]
files = [file for file in files if 'watershed' in file]
files = [file for file in files if 'marker' not in file]

# loop through saved point data and reformat to TIFs
for file in files:
    xr_data = xr.open_dataarray(image_dir + file)
    helper_functions.save_deepcell_tifs(xr_data, save_path=image_dir,  transform='watershed')


# average ensemble models together
avg_border = np.mean(data[:, 3, :, :, border_idx], axis=0)
avg_nuc = np.mean(data[:, 3, :, :, nuc_idx], axis=0)
io.imshow(avg_border)
avg_smoothed = nd.gaussian_filter(avg_nuc, 5)
io.imsave(os.path.join(image_dir, 'average_nucleus.tiff'), avg_nuc)
io.imsave(os.path.join(image_dir, 'average_nucleus_smoothed.tiff'), avg_smoothed)
io.imsave(os.path.join(image_dir, 'average_border.tiff'), avg_border)


plot_diff = data[1, 3, :, :, 1] - data[0, 3, :, :, 1]
fig, ax = plt.subplots()
mat = ax.imshow(plot_diff, cmap=plt.get_cmap('GnBu'))
fig.colorbar(mat)
fig.savefig(os.path.join(image_dir, 'interior_border_border_4_class_240k_max_class_examples_unbalanced_' + 'epoch_40vs30_nucleus.tiff'),
            dpi=300)
