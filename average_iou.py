# calculate average iou between cells
import skimage.io as io
import numpy as np
import skimage.measure

# DSB-score adapated from https://www.biorxiv.org/content/10.1101/580605v1.full
# object IoU matrix adapted from code written by Morgan Schwartz

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
ground_truth = io.imread(base_dir + "Point23/Nuclear_Interior_Mask.tif")
ground_truth = skimage.measure.label(ground_truth, connectivity=1)
predicted_data = io.imread(base_dir + 'analyses/20190606_params/mask_3_class_64_filters_256_densefilters_epoch_30.tiff')
predicted_data = skimage.measure.label(predicted_data, connectivity=1)


iou_matrix = np.zeros((np.max(ground_truth), np.max(predicted_data)))

for i in range(1, iou_matrix.shape[0] + 1):
    gt_img = ground_truth == i
    overlaps = np.unique(predicted_data[gt_img])
    for j in overlaps:
        pd_img = predicted_data == j
        intersect = np.sum(np.logical_and(gt_img, pd_img))
        union = np.sum(np.logical_or(gt_img, pd_img))
        # adjust index by one to account for not including background
        iou_matrix[i - 1, j - 1] = intersect / union


thresholds = np.arange(0.5, 0.96, 0.05)
scores = []

for i in range(len(thresholds)):

    iou_matrix_thresh = iou_matrix > thresholds[i]

    true_cells = iou_matrix_thresh.shape[0]
    pred_cells = iou_matrix_thresh.shape[1]

    # Calculate values based on projecting along prediction axis
    pred_proj = iou_matrix_thresh.sum(axis=1)
    # Zeros (aka absence of hits) correspond to true cells missed by prediction
    false_neg = np.count_nonzero(pred_proj == 0)
    # More than 2 hits corresponds to true cells hit twice by prediction, aka split
    split = np.count_nonzero(pred_proj >= 2)

    # Calculate values based on projecting along truth axis
    truth_proj = iou_matrix_thresh.sum(axis=0)
    # Empty hits indicate predicted cells that do not exist in true cells
    false_pos = np.count_nonzero(truth_proj == 0)
    # More than 2 hits indicates more than 2 true cells corresponding to 1 predicted cell
    merge = np.count_nonzero(truth_proj >= 2)

    # Ones are true positives excluding merge errors
    true_pos = np.count_nonzero(pred_proj == 1) - \
        (truth_proj[truth_proj >= 2].sum())


    score = true_pos / (true_pos + false_pos + false_neg)
    scores.append(score)

    # Calc dice jaccard stats for objects
    #dice, jaccard = get_dice_jaccard(iou_matrix_thresh)