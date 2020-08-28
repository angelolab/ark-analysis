import tempfile
import os

import numpy as np

from ark.utils import plot_utils
from skimage.draw import circle


def _generate_segmentation_labels(img_dims, num_cells=20):
    if len(img_dims) != 2:
        raise ValueError("must be image data of shape [rows, cols]")
    labels = np.zeros(img_dims, dtype="int16")
    radius = 20

    for i in range(num_cells):
        r, c = np.random.randint(radius, img_dims[0] - radius, 2)
        rr, cc = circle(r, c, radius)
        labels[rr, cc] = i

    return labels


def _generate_image_data(img_dims):
    if len(img_dims) != 3:
        raise ValueError("must be image data of [rows, cols, channels]")

    return np.random.randint(low=0, high=100, size=img_dims)


def test_plot_overlay():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    with tempfile.TemporaryDirectory() as temp_dir:
        # save with both tif and labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=example_images,
                                alternate_contour=None,
                                path=os.path.join(temp_dir, "example_plot1.tiff"))
        # save with just labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=None,
                                alternate_contour=None,
                                path=os.path.join(temp_dir, "example_plot2.tiff"))

        # save with two sets of labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=example_images,
                                alternate_contour=example_labels,
                                path=os.path.join(temp_dir, "example_plot3.tiff"))


def test_randomize_labels():
    labels = _generate_segmentation_labels((1024, 1024))
    randomized = plot_utils.randomize_labels(labels)

    assert np.array_equal(np.unique(labels), np.unique(randomized))

    # check that all pixels to belong to single cell in newly transformed image
    unique_vals = np.unique(labels)
    for val in unique_vals:
        coords = labels == val
        assert len(np.unique(randomized[coords])) == 1


def test_outline_objects():
    labels = _generate_segmentation_labels((1024, 1024), num_cells=300)
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


def test_plot_mod_ap():
    labels = ['alg1', 'alg2', 'alg3']
    thresholds = np.arange(0.5, 1, 0.1)
    mAP_array = [{'scores': [0.9, 0.8, 0.7, 0.4, 0.2]}, {'scores': [0.8, 0.7, 0.6, 0.3, 0.1]},
                 {'scores': [0.95, 0.85, 0.75, 0.45, 0.25]}]

    plot_utils.plot_mod_ap(mAP_array, thresholds, labels)


def test_plot_error_types():
    stats_dict = {
        'n_pred': 200,
        'n_true': 200,
        'correct_detections': 140,
        'missed_detections': 40,
        'gained_detections': 30,
        'merge': 20,
        'split': 10,
        'catastrophe': 20
    }

    stats_dict1 = {
        'n_pred': 210,
        'n_true': 210,
        'correct_detections': 120,
        'missed_detections': 30,
        'gained_detections': 50,
        'merge': 50,
        'split': 30,
        'catastrophe': 50
    }

    stats_dict2 = {
        'n_pred': 10,
        'n_true': 20,
        'correct_detections': 10,
        'missed_detections': 70,
        'gained_detections': 50,
        'merge': 5,
        'split': 3,
        'catastrophe': 5
    }

    plot_utils.plot_error_types([stats_dict, stats_dict1, stats_dict2], ['alg1', 'alg2', 'alg3'],
                                ['missed_detections', 'gained_detections', 'merge', 'split',
                                 'catastrophe'])
