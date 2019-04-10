import matplotlib.pyplot as plt
from PIL import Image
import skimage.measure
import numpy as np
import os

# code to take contoured data and generate mask necessary for CNN training

os.chdir('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point23/')

nuc_label = plt.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point23/Nuclear_Interior_Mask.tif')
mask = skimage.measure.label(nuc_label, connectivity=2)
x = Image.fromarray(np.uint16(mask))

x.save('Nuclear_Interior_Mask_Label.tif')


z = plt.imread('annotated/feature.tif')
