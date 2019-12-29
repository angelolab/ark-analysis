import numpy as np
import skimage.io as io
from skimage.transform import resize

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191224_Test_IF/Point14_BRCA/"

DNA = io.imread(base_dir + "DNA_rescaled.tiff")
Membrane = io.imread(base_dir + "Membrane_rescaled.tiff")

DNA_new = resize(DNA, [DNA.shape[0] / 2, DNA.shape[1] / 2], order=3).astype('int16')

Membrane_new = resize(Membrane, [Membrane.shape[0] / 2, Membrane.shape[1] / 2], order=3).astype('int16')

io.imsave(base_dir + "DNA_new.tif", DNA_new)
io.imsave(base_dir + "Membrane_new.tif", Membrane_new)