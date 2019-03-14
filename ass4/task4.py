import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
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


	# Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score


	# Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	# same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)

	# Step 4: Apply the mask to scores, boxes and classes

	np.random.seed(0)
	box_confidence = np.random.normal(size=(19, 19, 5, 1), loc=1, scale=4)
	boxes = np.random.normal(size=(19, 19, 5, 4), loc=1, scale=4)
	box_class_probs = np.random.normal(size=(19, 19, 5, 80), loc=1, scale=4)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
	print("scores[2] = " + str(scores[2]))
	print("boxes[2] = " + str(boxes[2]))
	print("classes[2] = " + str(classes[2]))
	print("scores.shape = " + str(scores.shape))
	print("boxes.shape = " + str(boxes.shape))
	print("classes.shape = " + str(classes.shape))

	return scores, boxes, classes


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
	scores -- tensor of shape (, None), predicted score for each box
	boxes -- tensor of shape (4, None), predicted box coordinates
	classes -- tensor of shape (, None), predicted class for each box

	Note: The "None" dimension of the output tensors has obviously to be less than max_boxes.
	Note also that this function will transpose the shapes of scores, boxes, classes.
	This is made for convenience.
	"""

	nms_indices = []
	# Use iou() to get the list of indices corresponding to boxes you keep

	# Use index arrays to select only nms_indices from scores, boxes and classes

	np.random.seed(0)
	scores = np.random.normal(size=(54,), loc=1, scale=4)
	boxes = np.random.normal(size=(54, 4), loc=1, scale=4)
	classes = np.random.normal(size=(54,), loc=1, scale=4)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
	print("scores[2] = " + str(scores[2]))
	print("boxes[2] = " + str(boxes[2]))
	print("classes[2] = " + str(classes[2]))
	print("scores.shape = " + str(scores.shape))
	print("boxes.shape = " + str(boxes.shape))
	print("classes.shape = " + str(classes.shape))

	return scores, boxes, classes


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

	### START CODE HERE ###

	# Retrieve outputs of the YOLO model (≈1 line)

	# Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)

	# Scale boxes back to original image shape.

	# Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)

	np.random.seed(0)
	yolo_outputs = (np.random.normal(size=(19, 19, 5, 1,), loc=1, scale=4),
	                np.random.normal(size=(19, 19, 5, 4,), loc=1, scale=4),
	                np.random.normal(size=(19, 19, 5, 80,), loc=1, scale=4))
	scores, boxes, classes = yolo_eval(yolo_outputs)
	print("scores[2] = " + str(scores[2]))
	print("boxes[2] = " + str(boxes[2]))
	print("classes[2] = " + str(classes[2]))
	print("scores.shape = " + str(scores.shape))
	print("boxes.shape = " + str(boxes.shape))
	print("classes.shape = " + str(classes.shape))

	return scores, boxes, classes


# VALIDATION
image = Image.open("test.jpg")
box_confidence = np.load("box_confidence.npy")
boxes = np.load("boxes.npy")
box_class_probs = np.load("box_class_probs.npy")
yolo_outputs = (box_confidence, boxes, box_class_probs)

image_shape = (720., 1280.)
out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape)

# Print predictions info
print('Found {} boxes'.format(len(out_boxes)))
# Draw bounding boxes on the image
draw_boxes(image, out_scores, out_boxes, out_classes)
# Display the results in the notebook
imshow()
