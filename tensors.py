'''
Important Study Material - https://github.com/mrdbourke/pytorch-deep-learning
'''

import torch

print("-----------------------------------------------------------------------------------------------------------------------------")
# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))
# Check for GPU (should return True)
print(torch.cuda.is_available())
# Check the version of torch and processor associated
print(torch.__version__)


print("-----------------------------------------------------------------------------------------------------------------------------")
## TENSORs and Types
scaler = torch.tensor(7)
print(f"scaler {scaler}")                   # output -> 7 of dtype tensor
print(scaler.item())                        # 7 of dtype python int
print(scaler.ndim)                          # no. of dimensions => scaler=0, vector=1 , matrix=2 and tensor=3+
print(scaler.shape)                         # size , no_of_elements
print(scaler.dtype)                         # torch.int64 (works same as numpy datatypes)

print("-----------------------------------------------------------------------------------------------------------------------------")
vector = torch.tensor([7,8,9])
print(f"vector {vector}")
print(vector.item)
print(vector.ndim)
print(vector.shape)
print(vector.dtype)

## the standard practice for nominclature is that the names for MATRIX and TENSOR should be in capitals(like constants)
## Bcoz the matrix and tensors are multidimensional meaning they have more than 2 dimensions hence CAPITAL 
## Same as Featues in ML or DL, they also can be many and hence can be multidimensional when denoted in array form

print("-----------------------------------------------------------------------------------------------------------------------------")
MATRIX = torch.tensor([[10,20],
                      [100,200]])
print(f'matrix {MATRIX}')
print(MATRIX.ndim)
print(MATRIX.shape)

print("-----------------------------------------------------------------------------------------------------------------------------")
TENSOR = torch.tensor([[[1,5,9]
                        ,[7,0,3],
                        [99,65,41],
                         [67,93,71]]])
print(f'tensor {TENSOR}')
print(TENSOR.ndim)
print(TENSOR.shape)

print("-----------------------------------------------------------------------------------------------------------------------------")
## Randomly Generated Tensors
scaler = torch.rand(3)
vector = torch.rand([3,2])
MATRIX = torch.rand([3,2,1])
TENSOR = torch.rand([3,2,1,1])

print(scaler, scaler.shape)
print(vector, vector.shape)
print(MATRIX, MATRIX.shape)
print(TENSOR, TENSOR.shape)