
import torch

if torch.backends.mps.is_available():
    print("MPS is available!")
else:
    print("MPS is not available. Please check your macOS version or PyTorch installation.")


x = torch.rand(3)
print(x)