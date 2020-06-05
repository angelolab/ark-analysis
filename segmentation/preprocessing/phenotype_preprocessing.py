import shutil
import os

from segmentation.utils import data_utils, io_utils

# TNBC preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/'

tnbc_dir = base_dir + '20200328_TNBC/fovs/new_good/'
fovs = data_utils.load_imgs_from_dir(tnbc_dir, imgs=['Beta catenin.tiff', 'Pan-Keratin.tiff',
                                                     'dsDNA.tiff', 'CD8.tiff', 'CD20.tiff',
                                                     'CD45.tiff', 'CD56.tiff'])

stitched_fovs = data_utils.stitch_images(fovs, 5)

for i in range(stitched_fovs.shape[-1]):
    io.imsave(tnbc_dir + stitched_fovs.channels.values[i] + '.tiff', stitched_fovs[0, :, :, i].values)


tifs = os.listdir('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200328_TNBC/fovs')
tifs = [tif for tif in tifs if '.tif' in tif]
tifs.sort()

# DCIS preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/'
fov_names = io_utils.list_folders(base_dir + 'Okay_Membrane')

for fov in fov_names:
    # shutil.copytree(os.path.join(base_dir, 'no_background', fov),
    #             os.path.join(base_dir, 'phenotyping_okay', fov))

    # CD45 needs to be denoised
    shutil.copy(os.path.join(base_dir, 'no_noise', fov, 'TIFs/CD45.tif'),
                os.path.join(base_dir, 'phenotyping_okay', fov, 'TIFs/CD45_denoised.tif'))

phenotype_data = data_utils.load_imgs_from_dir(base_dir + 'phenotyping_okay', img_sub_folder='TIFs',
                                               imgs=['CD45_denoised.tif', 'HH3.tif', 'PanKRT.tif',
                                                     'SMA.tif', 'ECAD.tif', 'CD44.tif'])
stitched_data = data_utils.stitch_images(phenotype_data, 5)
for i in range(stitched_data.shape[-1]):
    io.imsave(os.path.join(base_dir, 'phenotyping_okay', stitched_data.channels.values[i] + '.tiff'),
              stitched_data.values[0, :, :, i])
