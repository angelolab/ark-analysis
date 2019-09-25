import numpy as np
import skimage.measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io as io
import xarray as xr
import skimage.morphology as morph


# Perform watershed transformation over processed output files from deepcell

# read in relavant files
mask_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190917_naming/'

pixel_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_81_rf_512_dense_128_conv_epoch_18_processed.nc')
pixel_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_processed.nc')
watershed_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_watershed_81_rf_256_dense_64_conv_2erosion_epoch_03_processed.nc')

# point 1 melanoma
pixel_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_processed.nc')
watershed_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_watershed_81_rf_256_dense_64_conv_2erosion_epoch_03_processed.nc')

# point 1 melanoma HH3
pixel_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_HH3_81_rf_512_dense_128_conv_epoch_18_processed.nc')
watershed_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_HH3_watershed_81_rf_256_dense_64_conv_2erosion_epoch_04_processed.nc')

# point 1 melanoma
pixel_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_81_rf_512_dense_128_conv_epoch_24_3marker_processed.nc')
watershed_xr = xr.open_dataarray(mask_dir + 'Training_Freeze_1_Nuc_watershed_81_rf_256_dense_64_conv_2erosion_epoch_03_3marker_processed.nc')


points = pixel_xr.coords['points']
point = points.values[0]

# loop through points
for point in points:
    pixel_border = pixel_xr.loc[point, :, :, 'pixel_border'].values
    pixel_interior_smoothed = pixel_xr.loc[point, :, :, 'pixel_interior_smoothed'].values
    watershed_smoothed = watershed_xr.loc["point8", :, :, 'watershed_smoothed']

    # # use true masks to generate watershed seeds to check border/bg accuracy
    # seeds = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190823_TA489_Redo/zips/Point8_Cell_Mask_Label.tif')
    # seed_props = skimage.measure.regionprops(seeds)
    #
    # seed_array = np.zeros((1024, 1024))
    # for i in range(len(seed_props)):
    #     coords = np.floor(seed_props[i].centroid).astype('int')
    #     seed_array[coords[0], coords[1]] = 5
    #
    # maxs = seed_array > 1

    maxs = watershed_smoothed > 2
    interior_mask = pixel_interior_smoothed > 0.25

    # # calculate maxs from smoothed nuclear mask
    # maxs = peak_local_max(pixel_interior_smoothed, indices=False, min_distance=5)

    # # read in matlab generated mask
    # maxs = plt.imread(image_dir + '3_class_w_interior_and_watershed_epoch_20_maxs_matlab_0.05prominance_threshold.tiff')
    # maxs = maxs[:, :, 0]

    # use maxs to generate watershed
    markers = skimage.measure.label(maxs, connectivity=1)

    # remove any maxs that are 4 pixels or smaller
    for cell in np.unique(markers):
        mask = markers == cell
        if np.sum(mask) < 5:
            markers[mask] = 0

    # watershed over border mask vs negative interior mask?
    labels = np.array(watershed(-pixel_interior_smoothed, markers, mask=interior_mask, watershed_line=1))

    io.imsave(mask_dir + pixel_xr.name + 'point8_watershed_5_marker_label_mask.tiff', labels)

    # pixel expansion
    expanded = morph.dilation(labels, selem=morph.square(5))
    io.imsave(mask_dir + pixel_xr.name + "pixel_expansion_7.tiff", expanded)
