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
```
