import numpy as np
import copy
import skimage.morphology as morph
import skimage.measure
import math


# helper functions for everything else
def process_training_data(interior_contour, interior_border_contour):
    """Take in a contoured map of the border of each cell as well as entire cell,
    and generate annotated label map
    where each cell is its own unique pixel value

    Args:
        interior_contour: TIF with interior pixels as 1s, all else as 0
        interior_border_contour: TIF with all cellular pixels as 1s, all else as 0s

    Returns:
        label_contour: np.array with pixels belonging to each cell as a unique integer"""

    if np.sum(interior_contour) == 0:
        raise ValueError("Border contour is empty array")

    if np.sum(interior_border_contour) == 0:
        raise ValueError("Cell contour is empty array")

    if np.sum(interior_contour) > np.sum(interior_border_contour):
        raise ValueError("Arguments are likely switched, interior_contour is "
                         "larger than interior_border_contour")

    # label cells
    interior_contour = skimage.measure.label(interior_contour, connectivity=1)

    # for each individual cell, expand to slightly larger than original size one pixel at a time
    new_masks = copy.copy(interior_contour)
    for idx in range(2):
        for cell_label in np.unique(new_masks)[1:]:
            img = new_masks == cell_label
            img = morph.binary_dilation(img, morph.square(3))
            new_masks[img] = cell_label

    # set pixels to 0 anywhere outside bounds of original shape
    label_contour = randomize_labels(new_masks.astype("int"))
    label_contour[interior_border_contour == 0] = 0

    missed_pixels = np.sum(np.logical_and(interior_border_contour > 0, label_contour < 1))
    print("New Erosion and dilating resulted in a total of {} pixels "
          "that are no longer marked out of {}".format(missed_pixels,
                                                       interior_contour.shape[0] ** 2))

    return label_contour



def calc_adjacency_matrix(label_map, border_dist=0):
    """Generate matrix describing which cells are within the specified distance from one another

    Args
        label map: numpy array with each distinct cell labeled with a different pixel value
        border_dist: number of pixels separating borders of adjacent
            cells in order to be classified as neighbors

    Returns
        adjacency_matrix: numpy array of num_cells x num_cells with a 1 for neighbors
        and 0 otherwise"""

    if len(np.unique(label_map)) < 3:
        raise ValueError("array must be provided in labeled format")

    if not isinstance(border_dist, int):
        raise ValueError("Border distance must be an integer")

    adjacency_matrix = np.zeros((np.max(label_map) + 1, np.max(label_map) + 1), dtype='int')

    # We need to expand enough pixels such that cells which are within
    #  the specified border distance will overlap.
    # To find cells that are 0 pixels away, we expand 1 pixel in each direction to find overlaps
    # To check for cells that are 1 pixel away, we expand 2 pixels in either direction
    # we also need to factor in a center pixel which adds a constant of one
    morph_dist = border_dist * 2 + 3

    for cell in range(1, np.max(label_map) + 1):
        mask = label_map == cell
        mask = morph.dilation(mask, morph.square(morph_dist))
        overlaps = np.unique(label_map[mask])
        adjacency_matrix[cell, overlaps] = 1

    # reset background distance to 0
    adjacency_matrix[0, :] = 0
    adjacency_matrix[:, 0] = 0

    return adjacency_matrix

def euc_dist(coords_1, coords_2):
    """Calculate the euclidian distance between two y,x tuples

        Args
            coords_1: tuple of row, col values
            coords_2: tuple of row, col values

        Returns
            dist: distance between two points"""

    y = coords_1[0] - coords_2[0]
    x = coords_1[1] - coords_2[1]
    dist = math.sqrt(x**2 + y**2)
    return dist

def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells

        Args
            label_map: numpy array with unique cells given unique pixel labels

        Returns
            dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells"""

    if len(np.unique(label_map)) < 3:
        raise ValueError("Array must be provided in labeled format")

    cell_num = np.max(label_map)
    dist_matrix = np.zeros((cell_num + 1, cell_num + 1))
    props = skimage.measure.regionprops(label_map)

    for cell in range(1, cell_num + 1):
        cell_coords = props[cell - 1].centroid
        for tar_cell in range(1, cell_num + 1):
            tar_coords = props[tar_cell - 1].centroid
            dist = euc_dist(cell_coords, tar_coords)
            dist_matrix[cell, tar_cell] = dist

    return dist_matrix

