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
%matplotlib inline    #for notebook only 

image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)

image, label = dataset[700]
plt.imshow(image, cmap='gray')
print('Label:', label)
```

PyTorch doesn't know how to work with images. We need to convert the images into tensors. We can do this by specifying a transform while creating our dataset.
PyTorch datasets allow us to specify one or more transformation functions that are applied to the images as they are loaded. The `torchvision.transforms` module contains many such predefined functions. We'll use the `ToTensor` transform to convert images into PyTorch tensors.

```
import torchvision.transforms as transforms
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
img_tensor, label = dataset[0]
print(img_tensor.shape, label)
```
## Training and Validation Datasets
Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for validation. We can do this using the `random_spilt` method from PyTorch.
```
from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [50000, 10000])
```
## create data loaders to help us load the data in batches
`shuffle=True` for the training data loader to ensure that the batches generated in each epoch are different. This randomization helps generalize & speed up the training process. Since the validation data loader is used only for evaluating the model, there is no need to shuffle the images.
```
from torch.utils.data import DataLoader
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
```





