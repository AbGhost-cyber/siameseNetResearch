import random

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import math
from torchvision.datasets import ImageFolder
import torch.nn as nn

import matplotlib.pyplot as plt


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Define a function to plot the images
def plot_images(anchor_image, positive_image, negative_image):
    # Convert the tensors to numpy arrays
    anchor_image = anchor_image.numpy()
    positive_image = positive_image.numpy()
    negative_image = negative_image.numpy()

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(5, 5))

    # Plot the anchor image
    axs[0].imshow(np.transpose(anchor_image, (1, 2, 0)), cmap="gray")
    axs[0].set_title('Anchor Image')

    # Plot the positive image
    axs[1].imshow(np.transpose(positive_image, (1, 2, 0)), cmap="gray")
    axs[1].set_title('Positive Image')

    # Plot the negative image
    axs[2].imshow(np.transpose(negative_image, (1, 2, 0)), cmap="gray")
    axs[2].set_title('Negative Image')

    # Hide the axis labels
    for ax in axs:
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# Define a function to convert the tensors to numpy arrays
def convert_to_numpy(image_tensor):
    image_numpy = image_tensor.numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # Transpose to (H, W, C)
    return image_numpy


class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.targets = torch.tensor(self.targets)  # Convert targets to tensor for indexing

    def get_hard_triplet(self, anchor_index):
        anchor_label = self.targets[anchor_index]

        # Find positive samples with the same label as the anchor
        positive_indices = (self.targets == anchor_label).nonzero().flatten()
        positive_indices = positive_indices[positive_indices != anchor_index]

        # Find negative samples with different labels
        negative_indices = (self.targets != anchor_label).nonzero().flatten()

        # Select the hardest positive sample
        hardest_positive_index = self.get_hardest_sample(anchor_index, positive_indices)

        # Select a random negative sample
        hardest_negative_index = self.get_hardest_sample(anchor_index, negative_indices)

        # Load the images
        anchor_image_path = self.samples[anchor_index][0]
        positive_image_path = self.samples[hardest_positive_index][0]
        negative_image_path = self.samples[hardest_negative_index][0]

        anchor_image = Image.open(anchor_image_path)
        positive_image = Image.open(positive_image_path)
        negative_image = Image.open(negative_image_path)

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def get_hardest_sample(self, anchor_index, indices):
        anchor_path, _ = self.samples[anchor_index]
        anchor_features = Image.open(anchor_path)
        if self.transform is not None:
            anchor_features = self.transform(anchor_features)
        distances = []
        for index in indices:
            features, _ = self.samples[index]
            features = Image.open(features)
            if self.transform is not None:
                features = self.transform(features)
                distance = torch.norm(anchor_features - features)
                distances.append(distance.item())

        hardest_index = indices[distances.index(max(distances))]
        return hardest_index

    def __getitem__(self, index):
        return self.get_hard_triplet(anchor_index=index)

    def __len__(self):
        return len(self.samples)


class SimpleBranch(nn.Module):

    def __init__(self):
        super(SimpleBranch, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            # α = 10−4,β = 0.75 k = 2, n = 5
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


class SiameseNetwork(nn.Module):
    def __init__(self, branch):
        super(SiameseNetwork, self).__init__()
        self.branch = branch

    def forward(self, anchor_img, positive_img, negative_img):
        anchor_features = self.branch(anchor_img)
        positive_features = self.branch(positive_img)
        negative_features = self.branch(negative_img)

        return anchor_features, positive_features, negative_features


class TripletLoss(nn.Module):
    def __init__(self, margin=1.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean((distance_positive - distance_negative + self.margin).clamp(min=0))
        return loss


# std = 0.20561213791370392
# mean = 0.5613707900047302
mean = 0.2062
std = 0.1148
transformation = transforms.Compose([transforms.Resize((120, 120)),
                                     transforms.CenterCrop(120),
                                     ImageOps.invert,
                                     transforms.Grayscale(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[mean], std=[std])
                                     ])
transformation_test = transforms.Compose([transforms.Resize((120, 120)),
                                          transforms.CenterCrop(120),
                                          transforms.Grayscale(),
                                          transforms.ToTensor()
                                          ])
custom_dataset = CustomDataset(root="/Users/mac/research books/signature_research/data/faces/training/",
                               transform=transformation)
# Load the training dataset
train_dataloader = DataLoader(custom_dataset, shuffle=True, batch_size=64)

# pixel_values = []
#
# for i in range(len(custom_dataset)):
#     anchor_image, positive_image, negative_image = custom_dataset[i]
#
#     pixel_values.extend([anchor_image, positive_image, negative_image])
#
# pixel_values = torch.stack(pixel_values, dim=0)
# mean = torch.mean(pixel_values, dim=(0, 2, 3))
# std = torch.std(pixel_values, dim=(0, 2, 3))
#
# mean = mean.item()
# std = std.item()
#
# print("std", std)
# print("mean", mean)

simple_branch = SimpleBranch()
siamese_net = SiameseNetwork(branch=simple_branch)
loss_fn = TripletLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

counter = []
loss_history = []
iteration_number = 0
epoch = 20
# siamese_net.train()
# # Iterate through the epochs
# for epoch in range(epoch):
#
#     # Iterate over batches
#     for index, (anchor_image, positive_image, negative_image) in enumerate(train_dataloader, 0):
#
#         # Zero the gradients
#         optimizer.zero_grad()
#
#         anchor, positive, negative = siamese_net(anchor_image, positive_image, negative_image)
#
#         # Pass the outputs of the networks and label into the loss function
#         loss_contrastive = loss_fn(anchor, positive, negative)
#
#         # Calculate the backpropagation
#         loss_contrastive.backward()
#
#         # Optimize
#         optimizer.step()
#         scheduler.step(loss_contrastive)
#         # scheduler.step()
#
#         # Every 10 batches print out the loss
#         if index % 10 == 0:
#             print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
#             iteration_number += 10
#
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
#
# plt.plot(counter, loss_history)
# plt.show()
#
# torch.save(siamese_net.state_dict(), "hard_positive_simple_branch.pt")

test_model = SiameseNetwork(branch=simple_branch)
state_dict = torch.load('hard_positive_simple_branch.pt')
test_model.load_state_dict(state_dict)
test_model.eval()

# Locate the test dataset and load it into the SiameseNetworkDataset
custom_dataset1 = CustomDataset(root="/Users/mac/research books/signature_research/data/faces/testing/",
                                transform=transformation)
test_dataloader = DataLoader(custom_dataset1, batch_size=1, shuffle=True)

# Grab one image that we are going to test
dataiter = iter(test_dataloader)
anchor_image, _, _ = next(dataiter)

for i in range(15):
    # Iterate over 5 images and test them with the first image (x0)
    _, positive_image, negative_image = next(dataiter)

    # Concatenate the two images together
    concatenated_positive = torch.cat((anchor_image, positive_image, negative_image), 0)

    output_anchor, output_positive, output_negative = test_model(anchor_image, positive_image, negative_image)
    euclidean_distance_positive = torch.nn.functional.pairwise_distance(output_anchor, output_positive)
    euclidean_distance_negative = torch.nn.functional.pairwise_distance(output_anchor, output_negative)
    imshow(torchvision.utils.make_grid(concatenated_positive),
           f'Dissimilarity b/w Anchor and Positive: {euclidean_distance_positive.item():.2f}\n'
           f'Dissimilarity b/w Anchor and Negative: {euclidean_distance_negative.item():.2f}'
           )

if __name__ == '__main__':
    print()
