from dataloaders import mean, std
import torchvision
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

image = plt.imread("img/image.jpg")
image = to_tensor(image)
image = normalize(image.data, mean, std)
image = image.view(1, *image.shape)
image = nn.functional.interpolate(image, size=(256, 256))
vanilla_image = image

model = torchvision.models.resnet18(pretrained=True)
first_layer_out = model.conv1(image)
for layer in list(model.children())[:-2]:
	image = layer(image)
last_layer_out = image

filters = model.conv1.weight.data.numpy()
num_cols = 8
num_rows = 8
fig = plt.figure(figsize=(num_cols, num_rows))
for i in range(filters.shape[0]):
	ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
	fil = np.moveaxis(filters[i], 0, -1)
	fil = (1 / (2 * np.amin(fil))) * fil + 0.5
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
