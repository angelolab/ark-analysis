import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.io as io
import shutil

from segmentation.utils import data_utils

# extract cHL data
import h5py, re, os
import pandas as pd


base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200114_cHL/data/"
fname = 'cHL-MIF-Noah.20200114.h5'
f = h5py.File(base_dir + fname, 'r')

deidentified = pd.read_hdf(base_dir + fname, 'key')
deidentified[['label']].drop_duplicates()


def get_image(fname,image_id):
    return h5py.File(fname,'r')['images/'+image_id]


# extract TIFs
for i, r in deidentified.iloc[:, :].iterrows():
    img = np.array(get_image(base_dir + fname,r['image_id']))

    if r['label'] in ["CD3 (Opal 540)", "DAPI", "CD8 (Opal 540)", "CD4 (Opal 620)"]:
        if not os.path.isdir(base_dir + r['frame_id']):
            os.makedirs(base_dir + r['frame_id'])

        io.imsave(base_dir + r['frame_id'] + "/" + r['label'] + ".tiff", img.astype('float32'))

# combine CD4 and CD8
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200114_cHL/data/Good/"

good_dir = os.listdir(base_dir)
good_dir = [x for x in good_dir if ".DS" not in x]
for i in good_dir:
    if "CD4 (Opal 620).tiff" in os.listdir(base_dir + i):
        CD4 = io.imread(base_dir + i + "/CD4 (Opal 620).tiff")
        CD8 = io.imread(base_dir + i + "/CD8 (Opal 540).tiff")
        combined = CD4 + CD8
        io.imsave(base_dir + i + "/Membrane.tiff", combined)


base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200114_cHL/Great/"

fovs = os.listdir(base_dir + 'fovs')

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    if 'Membrane.tiff' in imgs:
        membrane_name = 'Membrane.tiff'
    else:
        membrane_name = 'CD3 (Opal 540).tiff'

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov], dtype='float32',
                                         imgs=['DAPI.tiff', membrane_name])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Tyler BRCA IF data
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191213_Tyler_BRCA/clean/"

fovs = os.listdir(base_dir + 'trim_borders')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'trim_borders'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DAPI.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Eliot data preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191219_Eliot/Good/"

fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'trim_borders'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DAPI.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# DCIS processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/'
fovs = os.listdir(os.path.join(base_dir, 'Great_Membrane/selected_fovs'))
fovs = [fov for fov in fovs if 'Point' in fov]

# copy files from no_bg folder to selected_no_bg folder so these can be used for training
for fov in fovs:
    original_folder = os.path.join(base_dir, 'Great_Membrane/selected_fovs', fov)
    new_folder = os.path.join(base_dir, 'Great_Membrane/no_bg_fovs', fov)
    os.makedirs(new_folder)
    imgs = os.listdir(original_folder)
    imgs = [img for img in imgs if '.tif' in img]

    for img in imgs:
        shutil.copy(os.path.join(base_dir, 'no_background', fov, 'TIFs', img), new_folder)

# load HH3 and whatever membrane marker is in each folder, crop to 512, save with consistent name
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/Great_Membrane/'
fovs = os.listdir(base_dir + 'no_bg_fovs')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'no_bg_fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'HH3.tif'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'no_bg_fovs'),
                                         fovs=[fov], imgs=['HH3.tif', membrane_channel])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# IMC 20191211 preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191211_IMC/Great/"

fovs = os.listdir(base_dir + 'fovs')
fovs = [point for point in fovs if "Point" in point]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['DNA.tiff', 'Membrane.tiff'])

    # only one crop per image since images are quite small: we'll center 512 in the FOV
    row_len = data.shape[1]
    col_len = data.shape[2]
    row_crop_start = math.floor((row_len - 512) / 2)
    col_crop_start = math.floor((col_len - 512) / 2)

    cropped_data = data.values[:, row_crop_start:(row_crop_start + 512),
                   col_crop_start:(col_crop_start + 512), :]

    folder = os.path.join(base_dir, 'cropped/{}_crop_0'.format(fov))
    os.makedirs(folder)
    io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[0, :, :, 0])
    io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[0, :, :, 1])


# IMC 20200120 preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200120_IMC/great/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(base_dir + 'fovs/' + fov)]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'Histone.tiff'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['Histone.tiff', membrane_channel])

    # some images are small than 512, others are only marginally bigger
    row_len, col_len = data.shape[1:3]
    new_data = np.zeros((1, max(512, row_len), max(512, col_len), 2), dtype='float32')

    # if either dimension is less than 512, we'll expand to 512
    new_data[:, :row_len, :col_len, :] = data.values

    # for dimensions that are only marginally larger than 512, we'll use center 512 crop
    if 512 < row_len < 768:
        row_crop_start = math.floor((row_len - 512) / 2)
        new_data = new_data[:, row_crop_start:(row_crop_start + 512), :, :]

    if 512 < col_len < 768:
        col_crop_start = math.floor((col_len - 512) / 2)
        new_data = new_data[:, :, col_crop_start:(col_crop_start + 512), :]

    cropped_data = data_utils.crop_image_stack(new_data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# 2019 CyCIF paper
# extract channels
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200209_CyCIF_SciRep/Tonsil-1/"
composite = io.imread(base_dir + "TONSIL-1_40X.ome.tif")

for chan in range(composite.shape[0]):
    io.imsave(base_dir + "Channel_{}.tif".format(chan + 1), composite[chan, :, :])

# generate crops
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200209_CyCIF_SciRep/Great/'
fovs = os.listdir(base_dir + 'renamed_channels')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'renamed_channels', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'renamed_channels'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DNA.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Roshan processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200219_Roshan/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov],
                                         imgs=['HH3.tif', 'CD138.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# melanoma preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200226_Melanoma/Great_Membrane/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov],
                                         imgs=['HH3.tif', 'NAKATPASE.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# IMC 20200411 preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/'
stacks = os.listdir(base_dir + 'full_stacks')
stacks = [stack for stack in stacks if '.tiff' in stack]
data_utils.split_img_stack(stack_dir=os.path.join(base_dir, 'full_stacks'),
                           output_dir=os.path.join(base_dir, 'fovs'),
                           stack_list=stacks,
                           indices=[0, 7, 17, 25, 32, 40],
                           names=['HH3.tiff', 'CK5.tiff', 'HER2.tiff', 'CD44.tiff',
                                  'ECAD.tiff', 'PanCK.tiff'])

# copy files into folders of 50 images each
all_fovs = os.listdir(base_dir + 'fovs')
all_fovs = [fov for fov in all_fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]
for folder_idx in range(math.ceil(len(all_fovs) / 50)):
    folder_path = os.path.join(base_dir, 'fovs/sub_folder_{}'.format(folder_idx))
    os.makedirs(folder_path)
    for fov in range(50):
        current_fov = all_fovs[folder_idx * 50 + fov]
        shutil.move(os.path.join(base_dir, 'fovs', current_fov),
                    os.path.join(base_dir, 'fovs', folder_path, current_fov))

# create stitched overlays of each to determine which markers will be included
folders = os.listdir(base_dir + 'fovs')
folders = [folder for folder in folders if 'sub' in folder]

for folder in folders:
    image_stack = data_utils.load_imgs_from_dir(base_dir + '/fovs/' + folder, variable_sizes=True,
                                                dtype='float32')
    stitched = data_utils.stitch_images(image_stack, 10)
    for img in range(stitched.shape[-1]):
        current_img = stitched[0, :, :, img].values
        io.imsave(os.path.join(base_dir, 'fovs', folder, stitched.channels.values[img] + '.tiff'),
                  current_img)

# after manual inspection, move selected FOVs in each channel sub-folder to same overall folder
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/fovs/'
fovs = os.listdir(base_dir + 'HER2')
fovs = [fov for fov in fovs if 'MB' in fov]

for fov in fovs:
    new_dir = os.path.join(base_dir, 'combined', fov)
    old_dir = os.path.join(base_dir, 'HER2', fov)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    shutil.copy(old_dir + '/HER2.tiff', new_dir + '/HER2.tiff')
    shutil.copy(old_dir + '/HH3.tiff', new_dir + '/HH312.tiff')


# after manual inspection to select best channel for each FOV, generate standard crops
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(base_dir + 'fovs/' + fov)]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'HH3.tiff'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['HH3.tiff', membrane_channel])

    # some images are small than 512, others are only marginally bigger
    row_len, col_len = data.shape[1:3]
    new_data = np.zeros((1, max(512, row_len), max(512, col_len), 2), dtype='float32')

    # if either dimension is less than 512, we'll expand to 512
    new_data[:, :row_len, :col_len, :] = data.values

    # for dimensions that are only marginally larger than 512, we'll use center 512 crop
    if 512 < row_len < 768:
        row_crop_start = math.floor((row_len - 512) / 2)
        new_data = new_data[:, row_crop_start:(row_crop_start + 512), :, :]

    if 512 < col_len < 768:
        col_crop_start = math.floor((col_len - 512) / 2)
        new_data = new_data[:, :, col_crop_start:(col_crop_start + 512), :]

    cropped_data = data_utils.crop_image_stack(new_data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# upsample labels
labels = os.listdir(base_dir)
labels = [label for label in labels if "Seg" in label]

for label_name in labels:
    label = io.imread(os.path.join(base_dir, label_name))

    label_resized = resize(label, [label.shape[0] * 2, label.shape[1] * 2], order=0, preserve_range=True)

    io.imsave(os.path.join(base_dir, label_name + "_Upsampled.tiff"), label_resized.astype('int16'))
