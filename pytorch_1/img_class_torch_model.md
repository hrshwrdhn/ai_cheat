# image classifier  Model
define our model.

A logistic regression model is almost identical to a linear regression model. It contains weights and bias matrices, and the output is obtained using simple matrix operations (pred = x @ w.t() + b).

As did in `linear regression`, we can use `nn.Linear` to create the model instead of manually creating and initializing the matrices.

Since `nn.Linear` expects each training example to be a vector, each 1x28x28 image tensor is flattened into a vector of size 784 (28*28) before being passed into the model.

The output for each image is a vector of size 10, with each element signifying the probability of a particular target label (i.e., 0 to 9). The predicted label for an image is simply the one with the highest probability.

```
import torch.nn as nn
input_size = 28*28
num_classes = 10
# Logistic regression model
model = nn.Linear(input_size, num_classes)
```

Visualize weight and bias
```
print(model.weight.shape)
model.weight

print(model.bias.shape)
model.bias
```

tranform image into 1-D array
`images.shape`  output=>   `torch.Size([128, 1, 28, 28])`   (batchsize is 128)`
```
images.reshape(128, 784)
```
 ## Let's extend the `nn.Module class` from PyTorch to define a custom model.
 
 ```
 class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()
 ```
 Inside the `__init__` constructor method, we instantiate the weights and biases using nn.Linear. 
 And inside the forward method, which is invoked when we pass a batch of inputs to the model, we flatten the input tensor and pass it into `self.linear`.
 `xb.reshape(-1, 28*28)` indicates to PyTorch that we want a view of the xb tensor with two dimensions.
 The length along the 2nd dimension is 28*28 (i.e., 784).
 One argument to `.reshape` can be set to -1 (in this case, the first dimension) to let PyTorch figure it out automatically based on the shape of the original tensor.
Note that the model no longer has `.weight` and `.bias` attributes (as they are now inside the `.linear` attribute), but it does have a `.parameters` method that returns a list containing the weights and bias.

```
for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
```

```
model.linear
```
output: ` Linear(in_features=784, out_features=10, bias=True)`
















