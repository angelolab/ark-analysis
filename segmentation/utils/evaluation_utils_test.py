import numpy as np

from segmentation.utils import evaluation_utils

import importlib
importlib.reload(evaluation_utils)


def test_calc_iou_matrix():
    true_labels = np.zeros((100, 100), 'int16')
    true_labels[5:15, 0:10] = 1
    true_labels[20:30, 20:30] = 2
    true_labels[50:60, 60:70] = 3

    predicted_labels = np.zeros((100, 100), 'int16')
    predicted_labels[5:15, 0:10] = 1
    predicted_labels[22:32, 22:32] = 2
    predicted_labels[90:95, 90:95] = 3

    iou = evaluation_utils.calc_iou_matrix(true_labels, predicted_labels)

    assert iou.shape == (3, 3)
    assert iou[0, 0] == 1

    intersect = np.sum(np.logical_and(true_labels == 2, predicted_labels == 2))
    union = np.sum(np.logical_or(true_labels == 2, predicted_labels == 2))
    assert iou[1, 1] == intersect / union

    assert iou[2, 2] == 0

    # multiple overlaps
    true_labels = np.zeros((100, 100), 'int16')
    true_labels[5:15, 0:10] = 1

    predicted_labels = np.zeros((100, 100), 'int16')
    predicted_labels[5:15, 0:5] = 1
    predicted_labels[5:15, 6:20] = 2

    iou = evaluation_utils.calc_iou_matrix(true_labels, predicted_labels)

    assert iou.shape == (1, 2)
    intersect1 = np.sum(np.logical_and(true_labels == 1, predicted_labels == 1))
    union1 = np.sum(np.logical_or(true_labels == 1, predicted_labels == 1))

    intersect2 = np.sum(np.logical_and(true_labels == 1, predicted_labels == 2))
    union2 = np.sum(np.logical_or(true_labels == 1, predicted_labels == 2))

    assert iou[0, 0] == intersect1 / union1
    assert iou[0, 1] == intersect2 / union2


def test_calc_modified_average_precision():
    test_iou = np.zeros((5, 5))
    test_iou[0, 0] = 0.9
    test_iou[1, 3] = 0.7
    test_iou[2, 2] = 0.6
    test_iou[3, 1] = 0.3
    test_iou[4, 4] = 0.2
    test_iou[4, 3] = 0.4

    thresholds = np.array((0.5, 0.6, 0.7, 0.8, 0.9))

    scores, false_positives, false_negatives = \
        evaluation_utils.calc_modified_average_precision(test_iou, thresholds)

    assert scores[0] == (3 / (3 + 2 + 2))
    assert scores[1] == (2 / (2 + 3 + 3))
    assert scores[2] == (1 / (1 + 4 + 4))
    assert scores[3] == (1 / (1 + 4 + 4))
    assert scores[4] == 0

    assert np.all(false_negatives[0] == [4, 5])
    assert np.all(false_positives[0] == [2, 5])

    assert np.all(false_negatives[2] == [2, 3, 4, 5])
    assert np.all(false_positives[2] == [2, 3, 4, 5])
