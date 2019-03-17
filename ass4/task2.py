import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes


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


def calculate_precision(num_tp, num_fp, num_fn):
	""" Calculates the precision for the given parameters.
		Returns 1 if num_tp + num_fp = 0

	Args:
		num_tp (float): number of true positives
		num_fp (float): number of false positives
		num_fn (float): number of false negatives
	Returns:
		float: value of precision
	"""
	if (num_fp + num_tp) == 0:
		return 1
	precision = (num_tp / (num_fp + num_tp))
	return precision


def calculate_recall(num_tp, num_fp, num_fn):
	""" Calculates the recall for the given parameters.
		Returns 0 if num_tp + num_fn = 0
	Args:
		num_tp (float): number of true positives
		num_fp (float): number of false positives
		num_fn (float): number of false negatives
	Returns:
		float: value of recall
	"""
	if (num_tp + num_fn) == 0:
		return 0
	recall = (num_tp / (num_tp + num_fn))
	return recall


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

	for gt in gt_boxes:
		iou = 0
		best_pred = None
		for p in prediction_boxes:
			local_iou = calculate_iou(p, gt)
			if local_iou >= iou_threshold and local_iou >= iou:
				iou = local_iou
				best_pred = p
		if best_pred is not None:
			matches.append([iou, best_pred, gt])

	# Sort all matches on IoU in descending order
	matches = sorted(matches, key=lambda x: x[0], reverse=True)

	return np.asarray([i[1] for i in matches]), np.asarray([i[2] for i in matches])


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
	"""Given a set of prediction boxes and ground truth boxes,
	   calculates true positives, false positives and false negatives
	   for a single image.
	   NB: prediction_boxes and gt_boxes are not matched!

	Args:
		prediction_boxes: (np.array of floats): list of predicted bounding boxes
			shape: [number of predicted boxes, 4].
			Each row includes [xmin, ymin, xmax, ymax]
		gt_boxes: (np.array of floats): list of bounding boxes ground truth
			objects with shape: [number of ground truth boxes, 4].
			Each row includes [xmin, ymin, xmax, ymax]
		iou_threshold: threshold value for iou
	Returns:
		dict: containing true positives, false positives, true negatives, false negatives
			{"true_pos": int, "false_pos": int, "false_neg": int}
	"""
	# Find best matches between prediction boxes and ground truth boxes.
	match_prediction_boxes, match_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

	# Compute true positives, false positives, false negatives
	true_positives = len(match_gt_boxes)
	false_negatives = len(gt_boxes) - true_positives
	false_positives = len(prediction_boxes) - true_positives

	return {"true_pos": true_positives, "false_pos": false_positives, "false_neg": false_negatives}


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
	"""Given a set of prediction boxes and ground truth boxes for all images,
	   calculates recall and precision over all images.

	   NB: all_prediction_boxes and all_gt_boxes are not matched!

	Args:
		all_prediction_boxes: (list of np.array of floats): each element in the list
			is a np.array containing all predicted bounding boxes for the given image
			with shape: [number of predicted boxes, 4].
			Each row includes [xmin, xmax, ymin, ymax]
		all_gt_boxes: (list of np.array of floats): each element in the list
			is a np.array containing all ground truth bounding boxes for the given image
			objects with shape: [number of ground truth boxes, 4].
			Each row includes [xmin, xmax, ymin, ymax]
		iou_threshold: threshold value for iou
	Returns:
		tuple: (precision, recall). Both float.
	"""
	true_positives = 0
	false_positives = 0
	false_negatives = 0
	for i, prediction_boxes in enumerate(all_gt_boxes):
		# Find matches between prediction boxes and ground truth boxes for each image.
		matches = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
		true_positives += matches["true_pos"]
		false_positives += matches["false_pos"]
		false_negatives += matches["false_neg"]
	# Calculate precision and recall.
	precision = calculate_precision(true_positives, false_positives, false_negatives)
	recall = calculate_recall(true_positives, false_positives, false_negatives)
	return precision, recall


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
	"""Given a set of prediction boxes and ground truth boxes for all images,
	   calculates the precision-recall curve over all images. Use the given
	   confidence thresholds to find the precision-recall curve.

	   NB: all_prediction_boxes and all_gt_boxes are not matched!

	Args:
		all_prediction_boxes: (list of np.array of floats): each element in the list
			is a np.array containing all predicted bounding boxes for the given image
			with shape: [number of predicted boxes, 4].
			Each row includes [xmin, xmax, ymin, ymax]
		all_gt_boxes: (list of np.array of floats): each element in the list
			is a np.array containing all ground truth bounding boxes for the given image
			objects with shape: [number of ground truth boxes, 4].
			Each row includes [xmin, xmax, ymin, ymax]
		confidence_scores: (list of np.array of floats): each element in the list
			is a np.array containting the confidence score for each of the
			predicted bounding box. Shape: [number of predicted boxes]

			E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
		iou_threshold: threshold value for iou

	Returns:
		tuple: (precision, recall). Both np.array of floats.
	"""
	confidence_thresholds = np.linspace(0, 1, 500)
	precision_array = []
	recall_array = []
	for threshold in confidence_thresholds:
		filtered_prediction_boxes = []
		for i, prediction_boxes in enumerate(all_prediction_boxes):
		# Only keep predictions that are higher than the confidence threshold.
			filtered_boxes = (np.array([prediction_box for j, prediction_box in enumerate(prediction_boxes)
										if np.greater_equal(confidence_scores[i][j], threshold)]))
			filtered_prediction_boxes.append(filtered_boxes)
		# Calculate precision and recall.
		precision, recall = calculate_precision_recall_all_images(filtered_prediction_boxes, all_gt_boxes,
																  iou_threshold)
		precision_array.append(precision)
		recall_array.append(recall)
	return np.array(precision_array), np.array(recall_array)


def plot_precision_recall_curve(precisions, recalls):
	"""Plots the precision recall curve.
		Save the figure to precision_recall_curve.png:
		'plt.savefig("precision_recall_curve.png")'

	Args:
		precisions: (np.array of floats) length of N
		recalls: (np.array of floats) length of N
	Returns:
		None
	"""
	plt.figure(figsize=(20, 20))
	plt.plot(recalls, precisions)
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.xlim([0.8, 1.0])
	plt.ylim([0.8, 1.0])
	plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precision_levels, recall_levels):
	""" Given a precision recall curve, calculates the mean average
		precision.

	Args:
		precision_levels: (np.array of floats) length of N
		recall_levels: (np.array of floats) length of N
	Returns:
		float: mean average precision
	"""
	recall = np.linspace(0.0, 1.0, 11)
	average_precision = 0
	for r in recall:
		max_precision = 0
		# Find indices of recall levels higher than r.
		rec = [index for index, value in enumerate(recall_levels) if value >= r]
		# Find corresponding precision values from the indices.
		pre = [precision_levels[j] for j in rec]
		# If list of precision values is not empty and max value in list is higher than the current max,
		# update max value.
		if pre:
			max_precision = max(pre)
		if not pre:
			max_precision = 0
		average_precision += max_precision
	average_precision /= 11
	return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
	""" Calculates the mean average precision over the given dataset
		with IoU threshold of 0.5

	Args:
		ground_truth_boxes: (dict)
		{
			"img_id1": (np.array of float). Shape [number of GT boxes, 4]
		}
		predicted_boxes: (dict)
		{
			"img_id1": {
				"boxes": (np.array of float). Shape: [number of pred boxes, 4],
				"scores": (np.array of float). Shape: [number of pred boxes]
			}
		}
	"""
	all_gt_boxes = []
	all_prediction_boxes = []
	confidence_scores = []

	for image_id in ground_truth_boxes.keys():
		pred_boxes = predicted_boxes[image_id]["boxes"]
		scores = predicted_boxes[image_id]["scores"]

		all_gt_boxes.append(ground_truth_boxes[image_id])
		all_prediction_boxes.append(pred_boxes)
		confidence_scores.append(scores)
	iou_threshold = 0.5
	precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
													 all_gt_boxes,
													 confidence_scores,
													 iou_threshold)
	plot_precision_recall_curve(precisions, recalls)
	mean_average_precision = calculate_mean_average_precision(precisions,
															  recalls)
	print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
	ground_truth_boxes = read_ground_truth_boxes()
	predicted_boxes = read_predicted_boxes()
	mean_average_precision(ground_truth_boxes, predicted_boxes)
