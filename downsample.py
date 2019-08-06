from skimage.transform import rescale, resize, downscale_local_mean
import skimage.io as io
import matplotlib.pyplot as plt
import os

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/Input_Data/Decidua_Object_Train_Initial'
output_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/Input_Data/Decidua_Object_Train_Initial_512'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

folders = os.listdir(base_dir)
folders = [folder for folder in folders if 'Point' in folder]
for folder in folders:
    os.makedirs(os.path.join(output_dir, folder))
    os.makedirs(os.path.join(output_dir, folder, 'annotated'))
    os.makedirs(os.path.join(output_dir, folder, 'raw'))

    label_image = os.listdir(os.path.join(base_dir, folder, 'annotated'))[1]
    original_tif = io.imread(os.path.join(base_dir, folder, 'annotated', label_image))
    modified_tif = resize(original_tif, (512, 512), order=0, anti_aliasing=False, preserve_range=True)
    io.imsave(os.path.join(output_dir, folder, 'annotated', label_image), modified_tif)

    raw_images = os.listdir(os.path.join(base_dir, folder, 'raw'))

    for image in raw_images:
        original_tif = io.imread(os.path.join(base_dir, folder, 'raw', image))
        modified_tif = resize(original_tif, (512, 512), order=0, anti_aliasing=False, preserve_range=True).astype('uint8')
        io.imsave(os.path.join(output_dir, folder, 'raw', image), modified_tif)



