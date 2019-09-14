import numpy as np
import os
import skimage.io as io
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import importlib
import helper_functions
importlib.reload(helper_functions)


# get directory where images are located
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses'
image_dir = base_dir + '/20190903_subsampling/'
plot_dir = image_dir + '/figs/'

# get names of each, clean up for subsequent saving
files = os.listdir(image_dir)
files = [file for file in files if 'npy' in file]
files = [file for file in files if 'erosion' in file]
#files = [file for file in files if 'interior_border_border_watershed_epoch' in file]
files.sort()

#prefix = files[0].split("interior_border_border")[0]
prefix = ''
names = files
names = [x.replace(prefix, '').replace('_metrics.npy', '') for x in names]


# load single point to get dimensions
temp = np.load(image_dir + files[0])
# load all data into a single numpy array
data = np.zeros(((len(files), ) + temp.shape), dtype='float32')

# axes on data: training run, image, x_dim, y_dim, output_mask
for i in range(len(files)):
    data[i, :, :, :, :] = np.load(os.path.join(image_dir, files[i]))

# save images back to folder for viewing
helper_functions.save_deepcell_tifs(data, names, image_dir, cohort=False, transform='watershed')


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
