import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
	"""Calculate intersection over union of single predicted and ground truth box.

	Args:
		prediction_box (np.array of floats): location of predicted object as
			[xmin, ymin, xmax, ymax]
		gt_box (np.array of floats): location of ground truth object as
			[xmin, ymin, xmax, ymax]

		returns:
			float: value of the intersection of union for the two boxes.
	"""

	# Return 0 if the boxes does not overlap
	if prediction_box[0] > gt_box[2] or gt_box[0] > prediction_box[2] or prediction_box[1] > gt_box[3] or gt_box[1] > \
			prediction_box[3]:
		return 0

	# Location of intersecting box
	I = [max(prediction_box[0], gt_box[0]),
	     max(prediction_box[1], gt_box[1]),
	     min(prediction_box[2], gt_box[2]),
	     min(prediction_box[3], gt_box[3])]

	# Area of intersection and union
	areaI = (I[2] - I[0]) * (I[3] - I[1])
	areaU = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1]) + (gt_box[2] - gt_box[
		0]) * (gt_box[3] - gt_box[1]) - areaI

	return areaI / areaU


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
	"""Finds all possible matches for the predicted boxes to the ground truth boxes.
		No bounding box can have more than one match.

		Remember: Matching of bounding boxes should be done with decreasing IoU order!

	Args:
		prediction_boxes: (np.array of floats): list of predicted bounding boxes
			shape: [number of predicted boxes, 4].
			Each row includes [xmin, ymin, xmax, ymax]
		gt_boxes: (np.array of floats): list of bounding boxes ground truth
			objects with shape: [number of ground truth boxes, 4].
			Each row includes [xmin, ymin, xmax, ymax]
	Returns the matched boxes (in corresponding order):
		prediction_boxes: (np.array of floats): list of predicted bounding boxes
			shape: [number of box matches, 4].
		gt_boxes: (np.array of floats): list of bounding boxes ground truth
			objects with shape: [number of box matches, 4].
			Each row includes [xmin, ymin, xmax, ymax]
	"""



	matches = []
	ious = [[0 for x in range(gt_boxes.shape[0])] for y in range(2)]
	i = 0

	# Find all possible matches with a IoU >= iou threshold
	for gt in gt_boxes:
		iou = 0
		ious[0][i] = i
		for p in prediction_boxes:
			local_iou = calculate_iou(p, gt)
			if local_iou >= iou_threshold and local_iou >= iou:
				matches.append(p)
				ious[1][i] = local_iou
				iou = local_iou
		i += 1

	# Sort all matches on IoU in descending order
	ious.sort(reverse=True)

	print('prediction: {}'.format(matches))
	print('gt: {}'.format(gt_boxes))
	print('threshold: {}'.format(iou_threshold))
	print('----------------------')

	# Find all matches with the highest IoU threshold
	return matches, gt_boxes
