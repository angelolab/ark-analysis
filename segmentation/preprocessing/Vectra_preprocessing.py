import skimage.io as io
import numpy as np

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191224_Test_IF/Vectra_Travis/"

Membrane_ED = io.imread(base_dir + "Membrane_C7_8_ED.tif")
Membrane_ED[Membrane_ED < 1.5] = 0
io.imsave(base_dir + "Membrane_ED.tif", Membrane_ED[6, :, :])

Nucleus_ED = io.imread(base_dir + "Nucleus_C6_8_ED.tif")
Nucleus_ED[Nucleus_ED < 1.5] = 0
io.imsave(base_dir + "Nucleus_ED.tif", Nucleus_ED[5, :, :])


Membrane_panc = io.imread(base_dir + "Membrane_C7_8_Pancreas.tif")
Membrane_panc[Membrane_panc < 5] = 0
io.imsave(base_dir + "Membrane_pancreas.tif", Membrane_panc[6, :, :])

Nucleus_panc = io.imread(base_dir + "Nucleus_C6_8_Pancreas.tif")
Nucleus_panc[Nucleus_panc < 1.2] = 0
io.imsave(base_dir + "Nucleus_pancreas.tif", Nucleus_panc[5, :, :])
