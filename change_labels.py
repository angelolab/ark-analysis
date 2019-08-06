# swap the labels for cell objects from whole-cell to nucleus for a subset of the cells
import numpy as np
import copy

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/Zips/'
image_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/20190615_Decidua/Point12/TIFs/'

all_labels = io.imread(base_dir + 'Point12_Objects_Mask_Label.tif')

cell_labels = io.imread(base_dir + 'Point12_Cell_Mask_Label.tif')

nuc_labels = io.imread(base_dir + 'Point12_Nuc_Mask_Label.tif')

vimentin = io.imread(image_dir + 'Vimentin.tif')
hlag = io.imread(image_dir + 'HLAG.tif')

switch_list = []

for cell in np.unique(all_labels):
    cell_mask = all_labels == cell
    vim_pos = np.sum(vimentin[cell_mask] > 0)
    hlag_pos = np.sum(hlag[cell_mask] > 0)

    if vim_pos / np.sum(cell_mask) > 0.4:
        pass

    elif hlag_pos / np.sum(cell_mask) > 0.4:
        pass
    else:
        switch_list.append(cell)

x = np.isin(all_labels, switch_list)

modified_labels = copy.copy(all_labels)
modified_labels[~x] = 0
io.imshow(modified_labels)

fig, ax = plt.subplots(2, 1)
ax[0].imshow(all_labels)
ax[1].imshow(modified_labels)

len(switch_list)

cell = switch_list[40]
fishy_count = 0

small_labels = copy.copy(all_labels)

for cell in switch_list:
    cell_mask = all_labels == cell
    overlap_id, overlap_count = np.array(np.unique(nuc_labels[cell_mask], return_counts=True))

    if len(overlap_id) == 1:
        # do nothing, only overlaps with nuclear contour, no need to change
        pass
    else:
        if overlap_id[0] == 0:
            overlap_id = overlap_id[1:]
            overlap_count = overlap_count[1:]

        max = np.max(overlap_count)
        id = overlap_id[overlap_count == max]
        small_labels[cell_mask] = 0
        small_labels[nuc_labels == id] = cell

io.imshow(small_labels)