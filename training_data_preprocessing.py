# preprocess label masks from hand contoured cells to generate training data
from skimage.morphology import disk
from skimage.morphology import binary_erosion, binary_dilation

total_cell = io.imread(base_dir + 'Point23/Nuclear_Interior_Border_Mask.tif')
interior_cell = io.imread(base_dir + 'Point23/Nuclear_Interior_Mask.tif')
interior_cell = binary_erosion(interior_cell, disk(1))
interior_cell = label(interior_cell)

new_masks = np.zeros((1024, 1024))
for cell_label in np.unique(interior_cell):
    # get the cell interior
    img = interior_cell == cell_label
    img = binary_dilation(img, disk(3))
    new_masks[img] = cell_label

plotting = randomize_labels(new_masks.astype("int"))
plotting[total_cell == 0] = 0
np.sum(np.logical_and(total_cell > 0, plotting < 1))
io.imshow(np.logical_and(total_cell > 0, plotting < 1))
io.imsave(base_dir + 'Point23/Nuclear_Interior_Border_Mask_Label.tif', plotting)
io.imsave(base_dir + 'Point23/Nuclear_Interior_Plotting_Mask.tif', interior_cell.astype('int16'))