# Gradient Descent and Linear Regression with PyTorch

## Introduction to Linear Regression
A model that predicts crop yields for apples and oranges (*target variables*) by looking at the average temperature, rainfall, and humidity (*input variables or features*) in a region. Here's the training data:

![linear-regression-training-data](https://i.imgur.com/6Ujttb4.png)

In a linear regression model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias :

```
yield_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
```
the yield of apples is a linear or planar function of temperature, rainfall and humidity:

![linear-regression-graph](https://i.imgur.com/4DJ9f8X.png)

#### Step1: Training data
Training data using two matrices: inputs and targets, each with one row per observation, and one column per variable.
```
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')
                   
# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
```
#### Step2: convert the arrays to PyTorch tensors.
```
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
```
#### Linear regression model from scratch

The weights and biases (`w11, w12,... w23, b1 & b2`) can also be represented as matrices, initialized as random values. 
The first row of `w` and the first element of `b` are used to predict the first target variable, i.e., yield of apples, and similarly, the second for oranges.
```
# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
```
#####  Note: 
`torch.randn` creates a tensor with the given shape, with elements picked randomly from a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with mean 0 and standard deviation 1.
*model* is simply a function that performs a matrix multiplication of the `inputs` and the weights `w` (transposed) and adds the bias `b` (replicated for each observation).

![matrix-mult](https://i.imgur.com/WGXLFvA.png)

#### Step4: define the model
```
def model(x):
    return x @ w.t() + b
```
`@` represents matrix multiplication in PyTorch, and the `.t` method returns the transpose of a tensor.
#### step5: Generate predictions
```
preds = model(inputs)
print(preds)
```
### Loss function
#### mean squared error (MSE)
To compare the model's predictions with the actual targets using  **mean squared error** (MSE).

* Calculate the difference between the two matrices (`preds` and `targets`).
* Square all elements of the difference matrix to remove negative values.
* Calculate the average of the elements in the resulting matrix.
```
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
```
Note: `torch.sum` returns the sum of all the elements in a tensor. The `.numel` method of a tensor returns the number of elements in a tensor.



#### Step5: Compute loss
```
loss = mse(preds, targets)
```

## Compute gradients

With PyTorch, we can automatically compute the gradient or derivative of the loss w.r.t. to the weights and biases because they have `requires_grad` set to `True`. 
```
gradient_wrt_w = w.grad

```
The gradients are stored in the `.grad` property of the respective tensors. Note that the derivative of the loss w.r.t. the weights matrix is itself a matrix with the same dimensions.

#### step 6 : Adjust weights and biases to reduce the loss
The loss is a quadratic function of our weights and biases, and our objective is to find the set of weights where the loss is the lowest.
the gradient indicates the rate of change of the loss, i.e., the loss function's slope



## the gradient descent_ optimization algorithm

If a gradient element is **positive**:

* **increasing** the weight element's value slightly will **increase** the loss
* **decreasing** the weight element's value slightly will **decrease** the loss

![postive-gradient](https://i.imgur.com/WLzJ4xP.png)

If a gradient element is **negative**:

* **increasing** the weight element's value slightly will **decrease** the loss
* **decreasing** the weight element's value slightly will **increase** the loss

![negative=gradient](https://i.imgur.com/dvG2fxU.png)

The increase or decrease in the loss by changing a weight element is proportional to the gradient of the loss w.r.t. that element. This observation forms the basis of _the gradient descent_ optimization algorithm.
Subtract from each weight element a small quantity proportional to the derivative of the loss w.r.t. that element to reduce the loss slightly.

#### Step 7: update weights and bias
```
 with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
 ```
#### step 8: verify that the loss is actually lower
```
loss = mse(preds, targets)
print(loss)
```
#### step 9 Reset the gradients to zero by invoking the  `.zero_()` method.
Before proceed, Reset the gradients to zero by invoking the `.zero_()` method. We need to do this because PyTorch accumulates gradients. Otherwise, the next time we invoke .backward on the loss, the new gradient values are added to the existing gradients, which may lead to unexpected results.
```
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)
```
Output: 
```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([0., 0.])
```
