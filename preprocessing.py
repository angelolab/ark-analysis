os.chdir("/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point18/")

nuc_label = plt.imread('Nuclear_Interior_Mask.tif')
mask = skimage.measure.label(nuc_label, connectivity=2)
x = Image.fromarray(np.uint16(mask))

x.save('Nuclear_Interior_Mask_Label.tif')
