# Working with Images in PyTorch

importing `torch` and `torchvision`. `torchvision` contains some utilities for working with image data. 
It also provides helper classes to download and import popular datasets like MNIST automatically.

 MNIST dataset consists of 28px by 28px grayscale images of handwritten digits (0 to 9) and labels for each image indicating which digit it represents. Here are some sample images from the dataset:

![mnist-sample](https://i.imgur.com/CAYnuo1.jpg)

```
# Imports
import torch
import torchvision
from torchvision.datasets import MNIST

# Download training dataset
dataset = MNIST(root='data/', download=True)  #len(dataset) is 60000

test_dataset = MNIST(root='data/', train=False)  #len(dataset) is 10000

dataset[0]
```
Output
```
(<PIL.Image.Image image mode=L size=28x28 at 0x7F2942D6F198>, 5)
```
It's a pair, consisting of a 28x28px image and a label. The image is an object of the class `PIL.Image.Image`, which is a part of the Python imaging library [Pillow](https://pillow.readthedocs.io/en/stable/). 
TO view the image within Jupyter using [`matplotlib`](https://matplotlib.org/), the de-facto plotting and graphing library for data science in Python.

The statement `%matplotlib inline` indicates to Jupyter that we want to plot the graphs within the notebook. Without this line, Jupyter will show the image in a popup. Statements starting with `%`are called magic commands and are used to configure the behavior of Jupyter itself. 

```
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)
```
PyTorch datasets allow us to specify one or more transformation functions that are applied to the images as they are loaded. The `torchvision.transforms` module contains many such predefined functions. We'll use the `ToTensor` transform to convert images into PyTorch tensors.
```
import torchvision.transforms as transforms
# MNIST dataset (images and labels)
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
img_tensor, label = dataset[0]
print(img_tensor.shape, label)
```
output:
```
torch.Size([1, 28, 28]) 5
```
The image is now converted to a 1x28x28 tensor. The first dimension tracks color channels. The second and third dimensions represent pixels along the height and width of the image, respectively. Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB). 



