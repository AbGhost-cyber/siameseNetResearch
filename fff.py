# import random
#
# lists = []
# for i in range(100):
#     should_get_same_class = random.randint(0, 1)
#     if should_get_same_class:
#         lists.append(should_get_same_class)
#
# if __name__ == '__main__':
#     print(len(lists))
# # root = "/Users/mac/research books/signature_research/data/faces/testing/"
import torch

from torch import nn
from torchvision.models import densenet121

shared_conv = nn.Sequential(
    nn.Conv2d(1, 28, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(28, 64, kernel_size=3, stride=2),
    nn.ReLU()
)
fc = nn.Sequential(
    nn.Linear(64 * 6 * 6, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

xx = torch.randn((1, 1, 28, 28))
res = shared_conv(xx)
result = res.view(res.size(0), -1)
output = fc(result)
print(res.shape)
print(output.shape)
if __name__ == '__main__':
    print()
