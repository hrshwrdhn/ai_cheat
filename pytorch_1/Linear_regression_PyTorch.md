# Linear regression using PyTorch built-ins

In deep learning, PyTorch provides several built-in functions and classes to make it easy to create and train models with just a few lines of code.
* `torch.nn` package from PyTorch contains utility classes for building neural networks.

```
import torch.nn as nn
```
* Represent the inputs and targets and matrices
```
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
```

## Dataset and DataLoader
`TensorDataset`, which allows access to rows from `inputs` and `targets` as tuples, and provides standard APIs for working with many different types of datasets in PyTorch.
```
from torch.utils.data import TensorDataset
# Define dataset
train_ds = TensorDataset(inputs, targets)

# check training dataset
print(train_ds[0:3])
```
Output: 
```
(tensor([[ 73.,  67.,  43.],
         [ 91.,  88.,  64.],
         [ 87., 134.,  58.]]), tensor([[ 56.,  70.],
         [ 81., 101.],
         [119., 133.]]))
```
The `TensorDataset` allows to access a small section of the training data using the array indexing notation (`[0:3]` in the above code). It returns a tuple with two elements. The first element contains the input variables for the selected rows, and the second contains the targets.
# data split in batches
create a DataLoader, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.
```
from torch.utils.data import DataLoader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
```
In each iteration, the data loader returns one batch of data with the given batch size. If shuffle is set to True, it shuffles the training data before creating batches. Shuffling helps randomize the input to the optimization algorithm, leading to a faster reduction in the loss.
```
# using the data loader in a for loop to look 
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break
```

#### check output 
```
tensor([[ 73.,  66.,  44.],
        [ 87., 134.,  58.],
        [ 68.,  97.,  70.],
        [103.,  43.,  36.],
        [102.,  43.,  37.]])
tensor([[ 57.,  69.],
        [119., 133.],
        [102., 120.],
        [ 20.,  38.],
        [ 22.,  37.]])

```
# nn.Linear

Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.
```
# Define model
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)
```
output
```
Parameter containing:
tensor([[0.5059, 0.5470, 0.1034],
        [0.0993, 0.5211, 0.5234]], requires_grad=True)
Parameter containing:
tensor([ 0.1677, -0.1366], requires_grad=True)
```
PyTorch models also have a helpful .parameters method, which returns a list containing all the weights and bias matrices present in the model. 
For this linear regression model,there is one weight matrix and one bias matrix.

# generate predictions 
```
preds = model(inputs)
```

# Loss Function
The `nn.functional` package contains many useful loss functions and several other utilities. 
```
# Import nn.functional
import torch.nn.functional as F
# Define loss function
loss_fn = F.mse_loss

# compute loss
loss = loss_fn(model(inputs), targets)
print(loss)
```

# Optimizer

Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD is short for "stochastic gradient descent". The term _stochastic_ indicates that samples are selected in random batches instead of as a single group.
```
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
```

# Train the model

We are now ready to train the model. We'll follow the same process to implement gradient descent:
1. Generate predictions
2. Calculate the loss
3. Compute gradients w.r.t the weights and biases
4. Adjust the weights by subtracting a small quantity proportional to the gradient
5. Reset the gradients to zero

The only change is that we'll work batches of data instead of processing the entire training data in every iteration. 
#### Define a utility function fit that trains the model for a given number of epochs
```
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```
Some things to note above:

    We use the data loader defined earlier to get batches of data for every iteration.

    Instead of updating parameters (weights and biases) manually, we use opt.step to perform the update and opt.zero_grad to reset the gradients to zero.

    We've also added a log statement that prints the loss from the last batch of data for every 10th epoch to track training progress. loss.item returns the actual value stored in the loss tensor.

Let's train the model for 100 epochs.

```
fit(100, model, loss_fn, opt, train_dl)

# Generate predictions
preds = model(inputs)

```



