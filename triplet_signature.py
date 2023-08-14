import random
from shutil import move

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import math
from torchvision.datasets import ImageFolder
import torch.nn as nn
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import RandomApply, RandomChoice, RandomRotation, RandomAffine
import torchvision.models as models


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# def select_images(source_folder, destination_folder, num_images_per_index=12):
#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # Iterate over the files in the source folder
#     for filename in os.listdir(source_folder):
#         if not filename.__contains__("png"):
#             continue
#         # Split the filename into parts
#         parts = filename.split('_')
#
#         # Extract the index and sub-index
#         index = int(parts[1])
#         sub_index = int(parts[2].split('.')[0])
#
#         # Check if the current index and sub-index meet the criteria
#         if sub_index <= num_images_per_index:
#             # Construct the source and destination paths
#             source_path = os.path.join(source_folder, filename)
#             destination_path = os.path.join(destination_folder, filename)
#
#             # Copy the image to the destination folder
#             shutil.copyfile(source_path, destination_path)
#
#
# # Specify the source and destination folders and the number of images per index
# full_org_folder = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_org'
# full_forg_folder = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_forg'
# new_full_org_folder = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_org'
# new_full_forg_folder = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_forg'

# # Select and copy the desired images for genuine signatures
# select_images(full_org_folder, new_full_org_folder)
#
# # Select and copy the desired images for forged signatures
# select_images(full_forg_folder, new_full_forg_folder)

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


class SignatureDataset(Dataset):
    def __init__(self, genuine_folder, forged_folder, transform=None):
        self.genuine_dataset = CustomDataset(genuine_folder, transform=transform)
        self.forged_dataset = CustomDataset(forged_folder, transform=transform)
        self.transform = transform

    def get_hard_triplet(self, anchor_index):
        anchor_image, positive_image, negative_image = self.genuine_dataset.get_hard_triplet(anchor_index)

        # Find the hardest negative sample from the forged dataset
        hardest_negative_index = self.forged_dataset.get_hardest_sample(anchor_index)

        # Load the images from the forged dataset
        negative_image_path = self.forged_dataset.samples[hardest_negative_index][0]
        negative_image = Image.open(negative_image_path)
        if self.transform is not None:
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def get_hardest_sample(self, anchor_index):
        return self.genuine_dataset.get_hardest_sample(anchor_index)

    def __getitem__(self, index):
        return self.get_hard_triplet(anchor_index=index)

    def __len__(self):
        return len(self.genuine_dataset)


class SignatureDataset2(Dataset):
    def __init__(self, genuine_folder, forged_folder, transform=None):
        self.genuine_dataset = CustomDataset(genuine_folder, transform=transform)
        self.forged_dataset = CustomDataset(forged_folder, transform=transform)
        self.transform = transform

    def get_hard_triplet(self, anchor_index):
        anchor_image, positive_image, negative_image = self.genuine_dataset.get_hard_triplet(anchor_index)

        # Find the hardest negative sample from the forged dataset
        # hardest_negative_index = self.get_hardest_sample(anchor_index, self.forged_dataset)

        # Load the images from the forged dataset
        negative_image_path = self.forged_dataset.samples[anchor_index][0]
        negative_image = Image.open(negative_image_path)
        if self.transform is not None:
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def get_hardest_sample(self, anchor_index, dataset):
        anchor_path, _ = dataset.samples[anchor_index]
        anchor_image = Image.open(anchor_path)
        if self.transform is not None:
            anchor_image = self.transform(anchor_image)

        # Extract features using a pre-trained model
        model = torchvision.models.resnet50(pretrained=True)
        # change to accept grayscale
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.eval()
        with torch.no_grad():
            anchor_features = model(torch.unsqueeze(anchor_image, 0))

            distances = []
            for index in range(len(dataset)):
                if index != anchor_index:
                    image_path, _ = dataset.samples[index]
                    image = Image.open(image_path)
                    if self.transform is not None:
                        image = self.transform(image)
                    features = model(torch.unsqueeze(image, 0))
                    distance = torch.norm(anchor_features - features)
                    distances.append(distance.item())

        hardest_index = distances.index(max(distances))
        return hardest_index

    def __getitem__(self, index):
        return self.get_hard_triplet(anchor_index=index)

    def __len__(self):
        return len(self.genuine_dataset)


# def organize_images_into_subfolders(root_folder):
#     for filename in os.listdir(root_folder):
#         if filename.endswith(".png"):
#             parts = filename.split("_")
#             original_index = int(parts[1])
#             folder_name = f"s{original_index}"
#             folder_path = os.path.join(root_folder, folder_name)
#
#             os.makedirs(folder_path, exist_ok=True)
#             file_path = os.path.join(root_folder, filename)
#             new_file_path = os.path.join(folder_path, filename)
#             move(file_path, new_file_path)
#
#
# original_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_org'
# forg_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_forg'
# # Usage example:
# organize_images_into_subfolders(original_signatures_root)
# organize_images_into_subfolders(forg_signatures_root)
# print("done")

std = 0.20561213791370392
mean = 0.5613707900047302
transformation = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.CenterCrop(40),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

original_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_org'
forg_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_forg'


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True))


class SimpleBranch(nn.Module):

    def __init__(self):
        super(SimpleBranch, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),
        )
        self.nin_block = nn.Sequential(
            nin_block(out_channels=128, kernel_size=2, strides=1, padding=0),
            nn.Dropout(p=0.3)
        )
        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.LazyLinear(2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.nin_block(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, branch):
        super(SiameseNetwork, self).__init__()
        self.branch = branch

    def forward(self, anchor_img, positive_img, negative_img):
        anchor_features = self.branch(anchor_img)
        positive_features = self.branch(positive_img)
        negative_features = self.branch(negative_img)

        return anchor_features, positive_features, negative_features

    def layer_summary(self, X_shape):
        return X_shape
        # X = torch.randn(*X_shape)
        # if self.branch.net is None:
        #     print("branch has not net attribute")
        # else:
        #     for layer in self.branch.net:
        #         X = layer(X)
        #         print(layer.__class__.__name__, 'output shape:\t', X.shape)


class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean((distance_positive - distance_negative + self.margin).clamp(min=0))
        return loss


train_dataset = SignatureDataset2(genuine_folder=original_signatures_root, forged_folder=forg_signatures_root,
                                  transform=transformation)
# Load the training dataset
batch_size = 64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
simple_branch = SimpleBranch()
siamese_net = SiameseNetwork(branch=simple_branch)
siamese_net.layer_summary((1, 1, 40, 40))
loss_fn = TripletLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.00003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

counter = []
loss_history = []
iteration_number = 0
epoch = 30
print(f"dataset size: {len(train_dataset)}")
print(f"batch per epoch {len(train_dataset) // batch_size}")
siamese_net.train()
# Iterate through the epochs
for epoch in range(epoch):

    # Iterate over batches
    for index, (anchor_image, positive_image, negative_image) in enumerate(train_dataloader, 0):

        # Zero the gradients
        optimizer.zero_grad()

        anchor, positive, negative = siamese_net(anchor_image, positive_image, negative_image)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = loss_fn(anchor, positive, negative)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()
        scheduler.step(loss_contrastive)
        # scheduler.step()

        # Every 10 batches print out the loss
        if (index + 1) % 10 == 0:
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

plt.plot(counter, loss_history)
plt.show()

torch.save(siamese_net.state_dict(), "triplet_sign.pt")

# example_batch = next(iter(train_dataloader))
# concatenated = torch.cat((example_batch[0], example_batch[1], example_batch[2]), 0)
# imshow(torchvision.utils.make_grid(concatenated))

# pixel_values = []
#
# with torch.no_grad():
#     for i in range(len(train_dataset)):
#         print(f"going for {i + 1}")
#         anchor_image, positive_image, negative_image = train_dataset[i]
#
#         pixel_values.append(anchor_image)
#         pixel_values.append(positive_image)
#         pixel_values.append(negative_image)
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

if __name__ == '__main__':
    print()
