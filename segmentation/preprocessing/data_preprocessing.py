## cHL data from DFCI
import h5py, re, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.io as io
import shutil

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200114_cHL/data/"
fname = 'cHL-MIF-Noah.20200114.h5'
f = h5py.File(base_dir + fname,'r')
print([x for x in f])

deidentified = pd.read_hdf(base_dir + fname,'key')
print(deidentified.shape)
deidentified.head()

deidentified[['label']].drop_duplicates()


def get_image(fname,image_id):
    return h5py.File(fname,'r')['images/'+image_id]

# extract TIFs

for i,r in deidentified.iloc[:, :].iterrows():
    img = np.array(get_image(base_dir + fname,r['image_id']))

    if r['label']  in ["CD3 (Opal 540)", "DAPI", "CD8 (Opal 540)", "CD4 (Opal 620)"]:
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



# Travis CODEX data
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191224_Test_IF/CODEX_Travis/"
Membrane1 = io.imread(base_dir + "Membrane1_rescaled.tif")
Membrane1[Membrane1 < 5000] = 0
Membrane2 = io.imread(base_dir + "Membrane2_rescaled.tif")

Membrane_combined = Membrane1 + Membrane2
io.imsave(base_dir + "Membrane_combined.tif", Membrane_combined)

Membrane_new = resize(Membrane, [Membrane.shape[0] / 2, Membrane.shape[1] / 2], order=3, preserve_range=True)
Membrane_new = Membrane_new.astype('int32')

io.imsave(base_dir + "DNA_resized.tif", DNA_new)
io.imsave(base_dir + "Membrane_resized.tif", Membrane_new)



# Tyler BRCA IF data

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191213_Tyler_BRCA/"

DNA = io.imread(base_dir + "Nucleus.tif")
Membrane = io.imread(base_dir + "Membrane.tif")

DNA_cropped = DNA[4200:5324, 2400:3424]
Membrane_cropped = Membrane[4200:5324, 2400:3424]
io.imsave(base_dir + "DNA_cropped.tif", DNA_cropped)
io.imsave(base_dir + "Membrane_cropped.tif", Membrane_cropped)


# Eliot data preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191219_Eliot/Great/"

points = os.listdir(base_dir)
points = [point for point in points if "Point" in point]

for point in points:
    DNA = io.imread(base_dir + point + "/DAPI.tif")
    Membrane = io.imread(base_dir + point + "/Membrane.tif")

    Membrane_resized = resize(Membrane, [Membrane.shape[0] * 2, Membrane.shape[1] * 2], order=3, preserve_range=True)
    DNA_resized = resize(DNA, [DNA.shape[0] * 2, DNA.shape[1] * 2], order=3, preserve_range=True)

    io.imsave(base_dir + point + "/DNA_Upsampled.tiff", DNA_resized.astype('int16'))
    io.imsave(base_dir + point + "/Membrane_Upsampled.tiff", Membrane_resized.astype('int16'))



# IMC preprocessing

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191211_IMC_PMC5791659/Great/raw/"

points = os.listdir(base_dir)
points = [point for point in points if "Point" in point]

for point in points:
    multi_tiff = io.imread(base_dir + point + "/{}.tiff".format(point.split("Point")[1]))
    multi_tiff = multi_tiff[20:]
    multi_tiff_smooth = nd.gaussian_filter(multi_tiff, 0.5)
    multi_tiff_resized = resize(multi_tiff, [multi_tiff.shape[0], multi_tiff.shape[1] * 3, multi_tiff.shape[2] * 3],
                                order=3, preserve_range=True)

    multi_tiff_resized = multi_tiff_resized.astype('int16')
    io.imsave(base_dir + point + "/DNA_Smoothed_Upsampled.tiff", multi_tiff_resized[3, :, :])
    io.imsave(base_dir + point + "/Membrane_Smoothed_Upsampled.tiff", multi_tiff_resized[0, :, :])

    io.imsave(base_dir + point + "/DNA.tiff", multi_tiff[3, :, :])
    io.imsave(base_dir + point + "/Membrane.tiff", multi_tiff[0, :, :])

# Vectra preprocessing
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


# 2020 IMC paper
base_dir = "/Users/noahgreenwald/Downloads/ome/"
composite_tifs = os.listdir(base_dir)
composite_tifs = [tif for tif in composite_tifs if ".tif" in tif]


for tif in composite_tifs:
    dir_name = os.path.splitext(tif)[0]
    save_path = os.path.join(base_dir, dir_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    composite = io.imread(base_dir + tif)
    io.imsave(os.path.join(save_path, "Histone.tiff"), composite[8, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "HER2.tiff"), composite[22, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "ER.tiff"), composite[27, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "PR.tiff"), composite[28, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "CD44.tiff"), composite[30, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "CD45.tiff"), composite[32, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "ECAD.tiff"), composite[37, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "EGFR.tiff"), composite[39, :, :].astype('float32'))
    io.imsave(os.path.join(save_path, "PanCK.tiff"), composite[45, :, :].astype('float32'))
    shutil.move(os.path.join(base_dir, tif), os.path.join(save_path, tif))

# 2019 CyCIF paper

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200209_CyCIF_SciRep/Tonsil-1/"

composite = io.imread(base_dir + "TONSIL-1_40X.ome.tif")

for chan in range(composite.shape[0]):
    io.imsave(base_dir + "Channel_{}.tif".format(chan + 1), composite[chan, :, :])


# Eliot test data preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20191219_Eliot/Test/"

stacks = os.listdir(base_dir)
stacks = [stack for stack in stacks if "stack" in stack]

for stack_name in stacks:
    stack = io.imread(os.path.join(base_dir, stack_name))
    DNA = stack[0, :, :]
    Membrane = stack[18, :, :]

    Membrane_resized = resize(Membrane, [Membrane.shape[0] * 2, Membrane.shape[1] * 2], order=3, preserve_range=True)
    DNA_resized = resize(DNA, [DNA.shape[0] * 2, DNA.shape[1] * 2], order=3, preserve_range=True)

    dir_name = os.path.join(base_dir, os.path.splitext(stack_name)[0])
    os.makedirs(dir_name)
    io.imsave(os.path.join(dir_name, "DNA_Upsampled.tiff"), DNA_resized.astype('int16'))
    io.imsave(os.path.join(dir_name, "Membrane_Upsampled.tiff"), Membrane_resized.astype('int16'))


# upsample labels
labels = os.listdir(base_dir)
labels = [label for label in labels if "Seg" in label]

for label_name in labels:
    label = io.imread(os.path.join(base_dir, label_name))

    label_resized = resize(label, [label.shape[0] * 2, label.shape[1] * 2], order=0, preserve_range=True)

    io.imsave(os.path.join(base_dir, label_name + "_Upsampled.tiff"), label_resized.astype('int16'))
