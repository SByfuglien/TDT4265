from dataloaders import mean, std
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
import matplotlib.pyplot as plt
import torch.nn as nn

image = plt.imread("img/image.jpg")
image = to_tensor(image)
image = normalize(image.data, mean, std)
image = image.view(1, *image.shape)
image = nn.functional.interpolate(image, size=(256, 256))

model = torchvision.models.resnet18(pretrained=True)
first_layer_out = model.conv1(image)

to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
torchvision.utils.save_image(to_visualize, "img/filters_first_layer.png")