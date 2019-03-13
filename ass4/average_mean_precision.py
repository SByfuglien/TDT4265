import numpy as np

recall_levels1 = [0.05, 0.1, 0.4, 0.7, 1.0][::-1]
precision_levels1 = [1.0, 1.0, 1.0, 0.5, 0.20][::-1]

recall_levels2 = [0.3, 0.4, 0.5, 0.7, 1.0][::-1]
precision_levels2 = [1.0, 0.80, 0.60, 0.5, 0.20][::-1]

recall_levels3 = [1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.2]
precision_levels3 = [0.5, 0.44, 0.5, 0.57, 0.5, 0.4, 0.5, 0.67, 1.0, 1.0]


def calculate_mean_average_precision(precision_levels, recall_levels):
	""" Given a precision recall curve, calculates the mean average
		precision.

	Args:
		precision_levels: (np.array of floats) length of N
		recall_levels: (np.array of floats) length of N
	Returns:
		float: mean average precision
	"""
	recall = np.linspace(0.0, 1.0, 11)[::-1]
	average_precision = 0
	max_precision = 0
	for i, r in enumerate(recall):
		# Find indices of recall levels higher than r.
		rec = [index for index, value in enumerate(recall_levels) if value >= r]
		# Find corresponding precision values from the indices.
		pre = [precision_levels[j] for j in rec]
		# If list of precision values is not empty and max value in list is higher than the current max,
		# update max value.
		if pre and max(pre) > max_precision:
			max_precision = max(pre)
		if not pre:
			max_precision = 0
		average_precision += max_precision
	average_precision /= 11
	return average_precision


m = calculate_mean_average_precision(precision_levels1, recall_levels1)
m2 = calculate_mean_average_precision(precision_levels2, recall_levels2)
print((m + m2) / 2)


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
	# YOUR CODE HERE
	raise NotImplementedError


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
		return 0
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
		iou_threshold: threshold value for iou.
	Returns the matched boxes (in corresponding order):
		prediction_boxes: (np.array of floats): list of predicted bounding boxes
			shape: [number of box matches, 4].
		gt_boxes: (np.array of floats): list of bounding boxes ground truth
			objects with shape: [number of box matches, 4].
			Each row includes [xmin, ymin, xmax, ymax]
	"""
	# Find all possible matches with a IoU >= iou threshold

	# Sort all matches on IoU in descending order

	# Find all matches with the highest IoU threshold
	raise NotImplementedError


def calculate_individual_image_result(
		prediction_boxes, gt_boxes, iou_threshold):
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
	# Find the bounding box matches with the highest IoU threshold

	# Compute true positives, false positives, false negatives
	false_positives = 0
	false_negatives = 0
	# Find best matches between prediction boxes and ground truth boxes.
	match_prediction_boxes, match_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
	true_positives = len(match_gt_boxes)
	# Find ground truth boxed with had no accurate prediction.
	no_match_gt_boxes = [gt_box for gt_box in gt_boxes if gt_box not in match_gt_boxes]
	for gt_box in no_match_gt_boxes:
		for prediction_box in prediction_boxes:
			# For each ground truth box that produced no match,
			# loop through all prediction boxes in order to determine false positives and false negatives.
			iou = calculate_iou(prediction_box, gt_box)
			if iou > iou_threshold:
				raise ValueError("iou calculation wrong, should be a true positive.")
			elif iou > 0:
				false_positives += 1
			elif iou == 0:
				false_negatives += 1
			else:
				raise ValueError("iou calculation wrong")
	return {"true_pos": true_positives, "false_pos": false_positives, "false_negatives": false_negatives}


def calculate_precision_recall_all_images(
		all_prediction_boxes, all_gt_boxes, iou_threshold):
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
	# Find total true positives, false positives and false negatives
	# over all images

	# Compute precision, recall
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


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
							   confidence_scores, iou_threshold):
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
	# Instead of going over every possible confidence score threshold to compute the PR
	# curve, we will use an approximation
	# DO NOT CHANGE. If you change this, the tests will not pass when we run the final
	# evaluation
	confidence_thresholds = np.linspace(0, 1, 500)
	precision_array = []
	recall_array = []
	for threshold in confidence_thresholds:
		# Only keep predictions that are higher than the confidence threshold.
		filtered_prediction_boxes = [prediction_box for i, prediction_box in enumerate(all_prediction_boxes)
									if np.greater_equal(confidence_scores[i], threshold)]
		# Calculate precision and recall.
		precision, recall = calculate_precision_recall_all_images(filtered_prediction_boxes, all_gt_boxes,
																  iou_threshold)
		precision_array.append(precision)
		recall_array.append(recall)
	return np.array(precision_array), np.array(recall_array)