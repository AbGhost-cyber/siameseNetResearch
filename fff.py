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

cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            # nn.Dropout(p=0.2),
            nn.ReLU(inplace=True)
        )

xx = torch.randn((1, 1, 224, 224))
print(cnn1(xx).shape)
if __name__ == '__main__':
    print()
