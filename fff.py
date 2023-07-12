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

cnn1 = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11),
    nn.ReLU(inplace=True),
    # nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(p=0.2),

    nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),
    nn.Dropout(p=0.2),

    nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
    nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
    nn.MaxPool2d(2, stride=2)
)

xx = torch.randn((1, 1, 145, 210))
print(cnn1(xx).shape)

if __name__ == '__main__':
    print()
