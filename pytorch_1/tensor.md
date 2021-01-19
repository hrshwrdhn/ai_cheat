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
