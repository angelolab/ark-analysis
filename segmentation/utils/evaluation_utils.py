import numpy as np
import skimage.measure
import pandas as pd
import os
import xarray as xr


def calc_iou_matrix(ground_truth_label, predicted_label):
    """Calculates pairwise ious between all cells from two masks

    Args:
        ground_truth_label: 2D label array representing ground truth contours
        predicted_label: 2D labeled array representing predicted contours

    Returns:
        iou_matrix: matrix of ground_truth x predicted cells with iou value for each
    """

    if len(ground_truth_label.shape) != 2 or len(predicted_label.shape) != 2:
        raise ValueError('input arrays must be two dimensional')

    iou_matrix = np.zeros((np.max(ground_truth_label), np.max(predicted_label)))

    for i in range(1, iou_matrix.shape[0] + 1):
        gt_img = ground_truth_label == i
        overlaps = np.unique(predicted_label[gt_img])
        overlaps = overlaps[overlaps > 0]
        for j in overlaps:
            pd_img = predicted_label == j
            intersect = np.sum(np.logical_and(gt_img, pd_img))
            union = np.sum(np.logical_or(gt_img, pd_img))

            # add values to matrix, adjust for background (0) not counted
            iou_matrix[i - 1, j - 1] = intersect / union
    return iou_matrix


def calc_mod_ap(iou_matrix, thresholds):
    """Calculates the average precision between two masks across a range of iou thresholds

    Args:
        iou_matrix: intersection over union matrix created by calc_iou_matrix function
        thresholds: list used to threshold iou values in matrix

    Returns:
        scores: list of modified average precision values for each threshold
        false_neg_idx: array of booleans indicating whether cell was
            flagged as false positive at each threshold
        false_pos_idx: array of booleans indicating whether cell was
            flagged as false negative at each threshold"""

    if np.any(np.logical_or(thresholds > 1, thresholds < 0)):
        raise ValueError("Thresholds must be between 0 and 1")

    scores = []
    false_negatives = []
    false_positives = []

    for i in range(len(thresholds)):
        # threshold iou_matrix as designated value
        iou_matrix_thresh = iou_matrix > thresholds[i]

        # Calculate values based on projecting along prediction axis
        pred_proj = iou_matrix_thresh.sum(axis=1)

        # Zeros (aka absence of hits) correspond to true cells missed by prediction
        false_neg = pred_proj == 0
        false_neg_count = np.sum(false_neg)
        false_neg_ids = np.where(false_neg)[0] + 1
        false_negatives.append(false_neg_ids)

        # Calculate values based on projecting along truth axis
        truth_proj = iou_matrix_thresh.sum(axis=0)

        # Empty hits indicate predicted cells that do not exist in true cells
        false_pos = truth_proj == 0
        false_pos_count = np.sum(false_pos)
        false_pos_ids = np.where(false_pos)[0] + 1
        false_positives.append(false_pos_ids)

        # Ones are true positives
        true_pos_count = np.sum(pred_proj == 1)

        score = true_pos_count / (true_pos_count + false_pos_count + false_neg_count)
        scores.append(score)

    return scores, false_positives, false_negatives


def compare_mod_ap(data_dict, thresholds):
    """Compare mAP across different paired true and predicted labels
    """

    y_true, y_pred = data_dict['y_true'], data_dict['y_pred']
    mod_ap_list = []

    for i in range(len(y_true)):
        iou_matrix = calc_iou_matrix(y_true[i], y_pred[i])
        scores, false_positives, false_negatives = calc_mod_ap(iou_matrix, thresholds)
        mod_ap_list.append({'scores': scores, 'false_pos': false_positives,
                            'false_neg': false_negatives})

    return mod_ap_list
