# take segmentation masks and TIFs and generate single cell data
import skimage.io as io
import skimage.morphology as morph
import skimage
import helper_functions
import importlib
importlib.reload(helper_functions)

# load segmentation masks
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
seg_folder = ''

seg_data = helper_functions.load_tifs_from_points_dir(seg_dir, seg_folder, ['Point23'], ['Nuclear_Mask_Label.tif'])
mask = seg_data.loc['Point23', 'Nuclear_Mask_Label.tif', :, :].values.astype('int')

mask_nuc = morph.erosion(mask, morph.square(3))
mask_whole = morph.dilation(mask_nuc, morph.square(5))

# load TIFs
tif_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
tif_folder = 'TIFsNoNoise'

image_data = helper_functions.load_tifs_from_points_dir(tif_dir, tif_folder, ['Point23'], ['dsDNA.tif', 'LaminAC.tif'])


# merge small cells?
cell_props = skimage.measure.regionprops(mask)
adjc_matrix = helper_functions.calc_adjacency_matrix(mask)

cell_arrays = np.zeros((3, marker_num, cell_num))
cell_arrays[0, :, :] = helper_functions.segment_data_function(mask_whole, image_data)
cell_arrays[1, :, :] = helper_functions.segment_data_function(mask_nuc)
cell_arrays[2, :, :] = helper_functions.segment_data_function(mask_cyto)

single_cell_xr = xr.DataArray(cell_arrays, coords=[['All_Signal', 'Nuclear_Signal', 'Cytoplasmic_Signal'],
                                                   image_data.channel.values, cell_ids],
                          dims=["point", "channel", "x_axis", "y_axis"])

