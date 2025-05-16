import torch

a = torch.tensor([-1,0.8,0.7,0.2,-0.1])
b = torch.nn.functional.softmax(a*10/2, dim=0)
print(b)