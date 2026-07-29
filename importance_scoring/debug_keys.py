import torch

text = torch.load("my_features/visual.pt")
print(len(text))
print(next(iter(text.values())).shape)
