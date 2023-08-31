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


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True))


# class SimpleBranch(nn.Module):
#
#     def __init__(self):
#         super(SimpleBranch, self).__init__()
#
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=2, stride=1),
#             nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),
#             nn.Dropout(p=0.3),
#
#             nn.Conv2d(64, 128, kernel_size=5, stride=1),
#             nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),
#             nn.Dropout(p=0.3),
#         )
#         self.nin_block = nn.Sequential(
#             nin_block(out_channels=128, kernel_size=2, strides=1, padding=0),
#             nn.Dropout(p=0.3)
#         )
#         self.fc_layer = nn.Sequential(
#             nn.LazyLinear(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#
#             nn.LazyLinear(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#
#             nn.LazyLinear(2)
#         )
#
#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = self.nin_block(x)
#         x = torch.flatten(x, 1)
#         x = self.fc_layer(x)
#         return x


# net = SimpleBranch()
# print(net(xx).shape)
cnn1 = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(p=0.3),

    nn.Conv2d(96, 256, kernel_size=5, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),
    nn.Dropout(p=0.3),

    nn.Conv2d(256, 1024, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
)

# Setting up the Fully Connected Layers
fc1 = nn.Sequential(
    nn.Linear(1024 * 11 * 11, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),

    nn.Linear(1024, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),

    nn.Linear(256, 2)
)
xx = torch.randn((1, 256, 256))
print(cnn1(xx).shape)
if __name__ == '__main__':
    print()
