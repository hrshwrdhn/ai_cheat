# Image classification.
We'll use the famous `MNIST` Handwritten Digits Database as training dataset. It consists of 28px by 28px grayscale images of handwritten digits (0 to 9) and labels for each image indicating which digit it represents. 
The `torchvision` package consists of popular datasets, model architectures, and common image transformations for computer vision.



```
import torch   
import torchvision
from torchvision.datasets import MNIST
# Download training dataset
dataset = MNIST(root='data/', download=True)
# create test data
test_dataset = MNIST(root='data/', train=False)
print(len(dataset),len(test_dataset))
```

Visualize image using `matplotlib`
```
import matplotlib
import matplotlib.pyplot as plt
import PIL
%matplotlib inline    #for notebook only 

image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)

image, label = dataset[700]
plt.imshow(image, cmap='gray')
print('Label:', label)

```
PyTorch doesn't know how to work with images. We need to convert the images into tensors. We can do this by specifying a transform while creating our dataset.

