import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Plotting data
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, idx):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        # if 1
        if should_get_same_class:
            while True:
                # Look until the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # Look until a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


root = "/Users/mac/research books/signature_research/data/faces/training/"
# Load the training dataset
folder_dataset = datasets.ImageFolder(root=root)

# Resize the images and transform to tensors
size = (100, 100)
transformation = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

# Initialize the dataset
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transformation)

# Create a simple dataloader just for visualization
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)


# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy().reshape(-1))


# print(example_batch[2].numpy().reshape(-1))
# imshow(torchvision.utils.make_grid(concatenated))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers=2, dropout_prob=0.2):
        super(VGGBlock, self).__init__()

        layers = []
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_prob))
            in_channels = out_channels

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


cnn1 = nn.Sequential(
    nn.Conv2d(1, 96, 11),  # size = [145,210]
    nn.ReLU(),
    nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
    nn.MaxPool2d(2, stride=2),  # size = [72, 105]

    nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
    nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
    nn.MaxPool2d(2, stride=2),  # size = [36, 52]
    nn.Dropout2d(p=0.3),

    nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
    nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
    nn.MaxPool2d(2, stride=2),  # size = [18, 26]
    nn.Dropout2d(p=0.3)
)

fc1 = nn.Sequential(
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.LazyLinear(out_features=1024),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 128)
)


x = torch.randn((1, 1, 224, 224))
output = cnn1(x)
output = fc1(output)
print(output.shape)


# print(y1.shape)


class SiameseNet2(nn.Module):
    def __init__(self):
        super(SiameseNet2, self).__init__()

        self.cnn1 = nn.Sequential(
            VGGBlock(1, 96),
            VGGBlock(96, 256),
            VGGBlock(256, 384),
            # VGGBlock(256, 512)
        )

        self.fc1 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=2)

        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # we pass in both images and obtain both vectors
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Dropout(p=0.2),

            nn.LazyConv2d(256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Dropout(p=0.2),

            nn.LazyConv2d(384, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),

            nn.LazyLinear(256),
            nn.ReLU(),

            nn.LazyLinear(2),
            nn.ReLU(),

        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class SigNet(nn.Module):
    '''
    Reference Keras: https://github.com/sounakdey/SigNet/blob/master/SigNet_v1.py
    '''

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # input size = [155, 220, 1]
            nn.Conv2d(1, 96, 11),  # size = [145,210]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [72, 105]
            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [36, 52]
            nn.Dropout(p=0.3),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),  # size = [18, 26]
            nn.Dropout(p=0.3),
            nn.Flatten(1, -1),  # 18*26*256
            nn.Linear(18 * 26 * 256, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
        )

        # TODO: init bias = 0

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        return x1, x2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance an contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=64)
net = SiameseNet2()
criterion = ContrastiveLoss()
lr = 5e-3
optimizer = optim.Adam(net.parameters(), lr=0.0005)

# counter = []
# loss_history = []
# iteration_number = 0
#
# num_epochs = 50
#
# net.train()
# # Iterate through the epochs
# for epoch in range(num_epochs):
#
#     # Iterate over batches
#     for i, (img0, img1, label) in enumerate(train_dataloader, 0):
#         # Zero the gradients
#         optimizer.zero_grad()
#
#         # Pass in the two images into the network and obtain two outputs
#         output1, output2 = net(img0, img1)
#
#         # Pass the outputs of the networks and label into the loss function
#         loss_contrastive = criterion(output1, output2, label)
#
#         # Calculate the backpropagation
#         loss_contrastive.backward()
#
#         # Optimize
#         optimizer.step()
#
#         # Every 10 batches print out the loss
#         if i % 10 == 0:
#             print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
#             iteration_number += 10
#
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
#
# show_plot(counter, loss_history)

# Locate the test dataset and load it into the SiameseNetworkDataset
# root = "/Users/mac/research books/signature_research/data/faces/testing/"
# folder_dataset_test = datasets.ImageFolder(root=root)
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test, transform=transformation)
# test_dataloader = DataLoader(siamese_dataset, batch_size=1, shuffle=True)
if __name__ == '__main__':
    print()
