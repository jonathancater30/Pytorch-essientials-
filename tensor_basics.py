import torch
import numpy as np  # we will use this later

#adjust data type like float, double, int, long
x=torch.ones(2,2,5, dtype=torch.double)

print(x.size())
print(x.dtype)

# Create a tensor with random values
x = torch.rand(5, 3)
y=torch.rand(5,3)

#add two tensors
print(x+y)
print(torch.add(x,y))

#add two tensors and store the result in a third tensor
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#multiply two tensors
print(x*y)
print(torch.mul(x,y))

#multiply two tensors and store the result in a third tensor
result=torch.empty(5,3)
torch.mul(x,y,out=result)
print(result)

# we can slice tensors
x=torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])
#if you have one element in your tensor then you can use the item() method to get the value as a Python number
print(x[1,1].item())

#Resizing: If you want to resize/reshape tensor, you can use torch.view:
x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)  # the size -1 is inferred from other dimensions
print(x.size(),y.size(),z.size())

#changing a tensor into a numpy array
a=torch.ones(5)
print(a)
b=a.numpy()
print(type(b))

#note: if you change the value of a, the value of b will also change due to the shared memory location unless of GPU
a.add_(1)
print(a)
print(b)

#changing a numpy array into a tensor
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#note: if you change the value of a, the value of b will also change due to the shared memory location unless of GPU


#requires_grad=True: This will track all the operations on the tensor. When you finish your computation you can call .backward() and have all the gradients computed automatically.
x=torch.ones(2,2,requires_grad=True)
print(x)
