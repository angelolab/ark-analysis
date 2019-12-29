import numpy as np
import os

import skimage.io as io
from skimage.transform import resize

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191224_Test_IF/PointC07_IMC/"

total_tiff = io.imread(base_dir + "C07.tiff")

total_tiff_resize = resize(total_tiff, [total_tiff.shape[0], total_tiff.shape[1] * 2, total_tiff.shape[2] * 2],
                           order=3)

io.imsave(base_dir + "DNA.tiff", total_tiff_resize[23, :, :])
io.imsave(base_dir + "Membrane.tiff", total_tiff_resize[21, :, :])

