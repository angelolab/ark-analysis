from segmentation.utils import plot_utils
from skimage.segmentation import find_boundaries
import skimage.io as io
import importlib

import numpy as np
importlib.reload(plot_utils)


def test_randomize_labels():
    labels = io.imread("segmentation/tests/test_output_files/segmentation_labels.tiff")
    randomized = plot_utils.randomize_labels(labels)

    assert np.array_equal(np.unique(labels), np.unique(randomized))

    # check that cell sizes are the same
    unique_vals = np.random.choice(np.unique(labels), 5)
    for val in unique_vals:
        coords = labels == val
        assert len(np.unique(randomized[coords])) == 1

def test_outline_objects():
    labels = io.imread("segmentation/tests/test_output_files/segmentation_labels.tiff")
    unique_vals = np.unique(labels)[1:]

    # generate a random subset of unique vals to be placed in each list
    vals = np.random.choice(unique_vals, size=60, replace=False)
    object_list = [vals[:20], vals[20:40], vals[40:]]

    outlined = plot_utils.outline_objects(labels, object_list)

    # check that cells in same object list were assigned the same label
    mask1 = np.isin(labels, object_list[0])
    assert len(np.unique(outlined[mask1])) == 1

    mask2 = np.isin(labels, object_list[1])
    assert len(np.unique(outlined[mask2])) == 1

    mask3 = np.isin(labels, object_list[2])
    assert len(np.unique(outlined[mask3])) == 1



