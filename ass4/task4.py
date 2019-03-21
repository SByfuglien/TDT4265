import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from drawing_utils import read_classes, draw_boxes, scale_boxes
from task2 import calculate_iou as iou


# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
	""" Filters YOLO boxes by thresholding on object and class confidence.

	Arguments:
		box_confidence -- np.array of shape (19, 19, 5, 1)
		boxes -- np.array of shape (19, 19, 5, 4)
		box_class_probs -- np.array of shape (19, 19, 5, 80)
		threshold -- real value, if [ highest class probability score < threshold],
			then get rid of the corresponding box

	Returns:
		scores -- np.array of shape (None,), containing the class probability score for selected boxes
		boxes -- np.array of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
		classes -- np.array of shape (None,), containing the index of the class detected by the selected boxes

	Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
	For example, the actual output size of scores would be (10,) if there are 10 boxes.
	"""

	# Step 1: Compute box scores
	box_scores = box_confidence * box_class_probs

	# Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
	box_classes = np.argmax(box_scores, axis=-1)
	box_class_scores = np.amax(box_scores, axis=-1)

	# Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	# same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
	confidence_mask = np.array(box_class_scores >= threshold)
	print(confidence_mask)
	# Step 4: Apply the mask to scores, boxes and classes
	scores = box_class_scores[confidence_mask]
	boxes = boxes[confidence_mask]
	classes = box_classes[confidence_mask]

	return scores, boxes, classes


def iou(prediction_box, gt_box):
	"""Calculate intersection over union of single predicted and ground truth box.

	Args:
		prediction_box (np.array of floats): location of predicted object as
			[xmin, ymin, xmax, ymax]
		gt_box (np.array of floats): location of ground truth object as
			[xmin, ymin, xmax, ymax]

		returns:
			float: value of the intersection of union for the two boxes.
	"""

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


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
	"""
	Applies Non-max suppression (NMS) to set of boxes

	Arguments:
		scores -- np.array of shape (None,), output of yolo_filter_boxes()
		boxes -- np.array of shape (None, 4), output of yolo_filter_boxes()
			that have been scaled to the image size (see later)
		classes -- np.array of shape (None,), output of yolo_filter_boxes()
		max_boxes -- integer, maximum number of predicted boxes you'd like
		iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

		Returns:
		scores -- np.array of shape (None,), containing the class probability score for selected boxes
		boxes -- np.array of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
		classes -- np.array of shape (None,), containing the index of the class detected by the selected boxes

	Returns:
	scores -- tensor of shape (, None), predicted score for each box
	boxes -- tensor of shape (4, None), predicted box coordinates
	classes -- tensor of shape (, None), predicted class for each box

	Note: The "None" dimension of the output tensors has obviously to be less than max_boxes.
	Note also that this function will transpose the shapes of scores, boxes, classes.
	This is made for convenience.
	"""
	best_box = []
	nms_indices = list(range(0, boxes.shape[0]))
	best_indices = []
	selection_not_found = True

	# Use iou() to get the list of indices corresponding to boxes you keep
	while selection_not_found:
		highest_score = -10000000000000000000000000000000000000000000000000000000000000
		for i in nms_indices:
			if scores[i] >= highest_score:
				highest_score = scores[i]
				best_box = boxes[i]
				index = i
		if index in best_indices:
			break
		best_indices.append(index)

		for i in nms_indices:
			if iou(boxes[i], best_box) >= iou_threshold:
				nms_indices.remove(i)
		if len(best_indices) == max_boxes:
			selection_not_found = False

	# Use index arrays to select only nms_indices from scores, boxes and classes
	selected_scores = scores[best_indices]
	selected_boxes = boxes[best_indices]
	selected_classes = classes[best_indices]

	return selected_scores, selected_boxes, selected_classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
	"""
	Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

	Arguments:
		yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 np.array:
						box_confidence: tensor of shape (None, 19, 19, 5, 1)
						boxes: tensor of shape (None, 19, 19, 5, 4)
						box_class_probs: tensor of shape (None, 19, 19, 5, 80)
		image_shape -- np.array of shape (2,) containing the input shape, in this notebook we use
			(608., 608.) (has to be float32 dtype)
		max_boxes -- integer, maximum number of predicted boxes you'd like
		score_threshold -- real value, if [ highest class probability score < threshold],
			then get rid of the corresponding box
		iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

	Returns:
		scores -- np.array of shape (None, ), predicted score for each box
		boxes -- np.array of shape (None, 4), predicted box coordinates
		classes -- np.array of shape (None,), predicted class for each box
	"""
	# Retrieve outputs of the YOLO model (≈1 line)
	box_confidence = yolo_outputs[0]
	boxes = yolo_outputs[1]
	box_class_probs = yolo_outputs[2]

	# Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

	# Scale boxes back to original image shape.
	boxes = scale_boxes(boxes, image_shape)

	# Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

	return scores, boxes, classes


# VALIDATION
np.random.seed(0)
BOX_CONFIDENCE = np.random.normal(size=(19, 19, 5, 1), loc=1, scale=4)
BOXES = np.random.normal(size=(19, 19, 5, 4), loc=1, scale=4)
BOX_CLASS_PROBS = np.random.normal(size=(19, 19, 5, 80), loc=1, scale=4)
SCORES, BOXES, CLASSES = yolo_filter_boxes(BOX_CONFIDENCE, BOXES, BOX_CLASS_PROBS, threshold=0.5)
print("scores[2] = " + str(SCORES[2]))
print("boxes[2] = " + str(BOXES[2]))
print("classes[2] = " + str(CLASSES[2]))
print("scores.shape = " + str(SCORES.shape))
print("boxes.shape = " + str(BOXES.shape))
print("classes.shape = " + str(CLASSES.shape))

np.random.seed(0)
SCORES = np.random.normal(size=(54,), loc=1, scale=4)
BOXES = np.random.normal(size=(54, 4), loc=1, scale=4)
CLASSES = np.random.normal(size=(54,), loc=1, scale=4)
SCORES, BOXES, CLASSES = yolo_non_max_suppression(SCORES, BOXES, CLASSES)
print("scores[2] = " + str(SCORES[2]))
print("boxes[2] = " + str(BOXES[2]))
print("classes[2] = " + str(CLASSES[2]))
print("scores.shape = " + str(SCORES.shape))
print("boxes.shape = " + str(BOXES.shape))
print("classes.shape = " + str(CLASSES.shape))

np.random.seed(0)
yolo_outputs = (np.random.normal(size=(19, 19, 5, 1,), loc=1, scale=4),
				np.random.normal(size=(19, 19, 5, 4,), loc=1, scale=4),
				np.random.normal(size=(19, 19, 5, 80,), loc=1, scale=4))
SCORES, BOXES, CLASSES = yolo_eval(yolo_outputs)
print("scores[2] = " + str(SCORES[2]))
print("boxes[2] = " + str(BOXES[2]))
print("classes[2] = " + str(CLASSES[2]))
print("scores.shape = " + str(SCORES.shape))
print("boxes.shape = " + str(BOXES.shape))
print("classes.shape = " + str(CLASSES.shape))

image = Image.open("test.jpg")
BOX_CONFIDENCE = np.load("box_confidence.npy")
BOXES = np.load("boxes.npy")
BOX_CLASS_PROBS = np.load("box_class_probs.npy")
yolo_outputs = (BOX_CONFIDENCE, BOXES, BOX_CLASS_PROBS)

image_shape = (720., 1280.)
out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape)

# Print predictions info
print('Found {} boxes'.format(len(out_boxes)))
# Draw bounding boxes on the image
draw_boxes(image, out_scores, out_boxes, out_classes)
plt.imshow(image)
plt.show()

