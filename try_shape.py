import torch

a = torch.tensor([[[1,2],[1,2]],[[1,2],[1,2]]])

print(a.shape)

print(a.sum((-1,-2,-3)))