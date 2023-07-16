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

cnn = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, stride=1),
    nn.ReLU(inplace=True)
)

shared_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
fc = nn.Sequential(
    nn.Linear(64 * 5 * 5, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

xx = torch.randn((64,1, 28, 28))
res = shared_conv(xx)
result = res.view(res.size(0), -1)
output = fc(result)
print(result.shape)
print(output.shape)
if __name__ == '__main__':
    print()
