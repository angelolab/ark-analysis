import numpy as np
import skimage.io as io
from skimage.transform import resize

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191224_Test_IF/Point14_BRCA/"

DNA = io.imread(base_dir + "Nucleus_rescaled.tif")
Membrane = io.imread(base_dir + "Membrane_rescaled.tif")

DNA_new = resize(DNA, [DNA.shape[0] / 2, DNA.shape[1] / 2], order=3, preserve_range=True)
DNA_new = DNA_new.astype("int32")

Membrane_new = resize(Membrane, [Membrane.shape[0] / 2, Membrane.shape[1] / 2], order=3, preserve_range=True)
Membrane_new = Membrane_new.astype('int32')

io.imsave(base_dir + "DNA_resized.tif", DNA_new)
io.imsave(base_dir + "Membrane_resized.tif", Membrane_new)