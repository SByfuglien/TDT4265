from dataloaders import mean, std
import torchvision
from torchvision.transforms.functional import to_tensor, normalize
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import json

image = plt.imread("img/image.jpg")
image = to_tensor(image)
image = normalize(image.data, mean, std)
image = image.view(1, *image.shape)
image = nn.functional.interpolate(image, size=(256, 256))
vanilla_image = image

model = torchvision.models.resnet18(pretrained=True)
# Send image through first convolutional layer and save activation
first_layer_out = model.conv1(image)

# Send image through first convolutional layer and save activation
for layer in list(model.children())[:-2]:
	image = layer(image)
last_layer_out = image

# Get weights from filters in first convolutional layer
filters = model.conv1.weight.data.numpy()
num_cols = 8
num_rows = 8
fig = plt.figure(figsize=(num_cols, num_rows))
for i in range(filters.shape[0]):
	ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
	fil = np.moveaxis(filters[i], 0, -1)
	# Normalize weights to [0, 1]
	fil = (1 / (2 * np.amax(abs(fil)))) * fil + 0.5
	ax1.imshow(fil)
	ax1.axis('off')
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
to_visualize2 = last_layer_out.view(last_layer_out.shape[1], 1, *last_layer_out.shape[2:])
torchvision.utils.save_image(to_visualize[:16], "img/filters_first_layer.png")
torchvision.utils.save_image(to_visualize2[:16], "img/filters_last_layer.png")

def loaded_model_analysis(history, history2=None, loss_bool=True, acc_bool=True):
	"""Plotting loss and accuracy from dictionary loaded from file."""
	# Loss history
	if loss_bool:
		plt.figure(figsize=(12, 8))
		plt.plot(history["loss"], label="Training loss")
		plt.plot(history["val_loss"], label="Validation loss")
		plt.plot(history["test_loss"], label="Test loss")
		plt.legend()
		plt.ylim(0, 1)
		plt.show()

	# Accuracy history
	if acc_bool:
		plt.figure(figsize=(12, 8))
		plt.plot(history["acc"], label="Training accuracy")
		plt.plot(history["val_acc"], label="Validation accuracy")
		plt.plot(history["test_acc"], label="Test accuracy")
		plt.legend()
		# plt.ylim(0.7, 1)
		plt.show()

	if history2 is not None:
		plt.figure(figsize=(12, 8))
		plt.plot(history["loss"], label="Training loss")
		plt.plot(history["val_loss"], label="Validation loss")
		plt.plot(history2["loss"][0::3], label="Resnet18 Training loss")
		plt.plot(history2["val_loss"][0::3], label="Resnet18 Validation loss")
		plt.legend()
		# plt.ylim(0, 1)

with open('models/history.json', 'r') as F:
	HISTORY = json.loads(F.read())
with open('models/history_transfer_learning.json', 'r') as F:
	HISTORY_2 = json.loads(F.read())
loaded_model_analysis(HISTORY, HISTORY_2)