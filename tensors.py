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
print(scaler.ndim)                          # no. of dimensions => scaler=0, vector=1 , matrix=2 and tensor=3
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

torch.rand([3,4,2])


