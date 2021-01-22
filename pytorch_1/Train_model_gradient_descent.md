# Train the model using gradient descent
To reduce the loss and improve model using the gradient descent optimization algorithm, train_ the model using the following steps:

1. Generate predictions
2. Calculate the loss
3. Compute gradients w.r.t the weights and biases
4. Adjust the weights by subtracting a small quantity proportional to the gradient
5. Reset the gradients to zero

```
# Generate predictions
preds = model(inputs)

# Calculate the loss
loss = mse(preds, targets)

# Compute gradients
loss.backward()

# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

```

# Train for multiple epochs
```
# Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
```

## Functions: 
```
def model(x):
    return x @ w.t() + b
# @ represents matrix multiplication in PyTorch, and the .t method returns the transpose of a tensor.
# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

```


