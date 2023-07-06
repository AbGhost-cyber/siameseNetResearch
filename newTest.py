from math import floor

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torchvision import models
from torch.nn import init
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

torch.manual_seed(1)


def accuracy(distances, y, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0
    same_id = (y == 1)

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d + step, step):
        true_positive = (distances <= threshold_d) & (same_id)
        true_positive_rate = true_positive.sum().float() / same_id.sum().float()
        true_negative = (distances > threshold_d) & (~same_id)
        true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

        acc = 0.5 * (true_negative_rate + true_positive_rate)
        max_acc = max(max_acc, acc)
    return max_acc


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

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Look untill a different class image is found
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


# Load the training and testing dataset
folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/training/")
folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((120, 120)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.RandomRotation(25),
                                     transforms.RandomCrop((100, 100)),
                                     transforms.ToTensor()
                                     ])

# Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation)

siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                             transform=transformation)

# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, and also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)


# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy().reshape(-1))


# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class CosineLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(CosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_distance = F.cosine_similarity(output1, output2, dim=1)
        loss_cosine = torch.mean((1 - label) * torch.pow(cosine_distance, 2) +
                                 label * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        return loss_cosine


# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

if __name__ == '__main__':
    print()


class TestSign(nn.Module):

    def __init__(self):
        super(TestSign, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=384, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=384 * 9 * 9, out_features=1024),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=128)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


print(f"Using device: {device}")

net = TestSign().to(device)
# Define your margin schedule
min_margin = 3
max_margin = 1
num_epochs = 50

# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=64)
print(len(train_dataloader))
optimizer = optim.Adam(net.parameters(), lr=0.002)
criterion = ContrastiveLoss().to(device)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
margin_schedule = [min_margin + (max_margin - min_margin) * (epoch / (num_epochs - 1)) for epoch in range(num_epochs)]

counter = []
loss_history = []
iteration_number = 0

for epoch in range(num_epochs):

    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):

        # Send the images and labels to device
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2 = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()
        # lr_scheduler.step()

        # # Adjust margin based on epoch
        # criterion.margin = margin_schedule[epoch]

        # Every 20 batches print out the loss
        if i % 20 == 0:
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# plt.plot(loss_history, label="Loss")
# plt.show()


siamese_dataset1 = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                         transform=transformation)
test_dataloader = DataLoader(siamese_dataset1, batch_size=1, shuffle=True)

# Grab one image that we are going to test
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    # Iterate over 5 images and test them with the first image (x0)
    _, x1, label2 = next(dataiter)

    # Concatenate the two images together
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(x0.to(device), x1.to(device))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')

# x_value = torch.randn((1, 1, 100, 100))
# res = cnn1(x_value)
# batch_size = res.size(0)
# flattened = res.view(batch_size, -1)
# res_final = fc1(flattened)
# flattened_size = 128 * 10 * 10
# input_width = 96
# input_height = 96
# kernel_size = 3
# stride = 2
# output_width = floor((input_width - (kernel_size - 1) - 1) / stride + 1)
if __name__ == '__main__':
    print()
    # print(res.shape)
    # print(res_final.shape)
    # print((96 - 2 - 1) / 2 + 1)
