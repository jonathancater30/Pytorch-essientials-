import torch 


x=torch.randn(3,requires_grad=True)
print(x)

y= x+2
print(y)

z=y*y*2
print(z)

z=z.mean()
print(z)


#to calculate the gradients
z.backward()
print(x.grad)

#grad can be implicitly created only for scalar outputs

#to stop PyTorch from tracking history on Tensors with requires_grad=True, you can wrap the code block in with torch.no_grad():
#x.requires_grad_(False) or x.detach()
#with torch.no_grad():
#recall the trailing _ in PyTorch signifies that the method is performed in-place

weights = torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()