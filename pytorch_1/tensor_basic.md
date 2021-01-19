# Pytorch basics

## import the torch module 
```
import torch
```
## Tensors
Tensor is a multi-dimensional matrix containing elements of a single data type

**Torch** defines 10 tensor types with CPU and GPU variants which are as follows: 

---

| **Data type | dtype | CPU tensor | GPU tensor** |
| ----------- | ----------- |----------- | ----------- |
|32-bit floating point | torch.float32 or torch.float | torch.FloatTensor | torch.cuda.FloatTensor|
|64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor|
|16-bit floating point 1 | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor|
|16-bit floating point 2 | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor|
|32-bit complex | torch.complex32 |
|64-bit complex | torch.complex64 |
|128-bit complex |  torch.complex128 or torch.cdouble |	
|8-bit integer (unsigned) |  torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |
|8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor | 
|32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor |
|64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor |
| Boolean | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |

### PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array.
A tensor with a single number
```
t1 = torch.tensor(4.)

```
**OUTPUT**
```
tensor(4.)
```
Checking the dtype attribute of our tensor.
```
t1.dtype
```
**OUTPUT**
```
torch.float32
```
others example
```
# Vector
>>> torch.tensor([1., 2, 3, 4])
tensor([1., 2., 3., 4.])

# Matrix
>>> torch.tensor([[1., -1.], [1., -1.]])
tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])
        
>>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
        
#  3-dimensional array
>>> torch.tensor([
    [[11, 12, 13], 
     [13, 14, 15]], 
    [[15, 16, 17], 
     [17, 18, 19.]]])
tensor([[[11., 12., 13.],
         [13., 14., 15.]],

        [[15., 16., 17.],
         [17., 18., 19.]]])
```
Note: *torch.tensor()* always copies data. If you have a Tensor data and just want to change its *requires_grad* flag, use *requires_grad_()* or *detach()* to avoid a copy. If you have a numpy array and want to avoid a copy, use *torch.as_tensor()*

### Shape
To inspect the length along each dimension using the **.shape** property of a tensor.
```
>>> print(t2)
>>>t2.shape

tensor([1., 2., 3., 4.])
torch.Size([4])

>>>print(t4)
>>>t4.shape

tensor([[[11., 12., 13.],
         [13., 14., 15.]],

        [[15., 16., 17.],
         [17., 18., 19.]]])

torch.Size([2, 2, 3])
```
Note: it's not possible to create tensors with an improper shape

# Tensor functions

**Create a tensor with a fixed value for every element**
```
torch.full((3, 2), 42)
```
tensor([[42, 42],
        [42, 42],
        [42, 42]])
        
**Concatenate two tensors with compatible shapes**
```
t7 = torch.cat((t3, t6))
t7
```
tensor([[ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [42., 42.],
        [42., 42.],
        [42., 42.]])
        
        
**Compute the sin of each element**
```
t8 = torch.sin(t7)
t8
```
tensor([[-0.9589, -0.2794],
        [ 0.6570,  0.9894],
        [ 0.4121, -0.5440],
        [-0.9165, -0.9165],
        [-0.9165, -0.9165],
        [-0.9165, -0.9165]])

**Change the shape of a tensor**
```
t9 = t8.reshape(3, 2, 2)
t9
```
tensor([[[-0.9589, -0.2794],
         [ 0.6570,  0.9894]],
        [[ 0.4121, -0.5440],
         [-0.9165, -0.9165]],
        [[-0.9165, -0.9165],
         [-0.9165, -0.9165]]])


