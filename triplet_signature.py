import random
from shutil import move

import numpy as np
import torch
import torchvision
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# class CustomDataset(ImageFolder):
#     def __init__(self, root, transform=None):
#         super().__init__(root, transform=transform)
#         self.targets = torch.tensor(self.targets)  # Convert targets to tensor for indexing
#
#     def get_hard_triplet(self, anchor_index):
#         anchor_label = self.targets[anchor_index]
#
#         # Find positive samples with the same label as the anchor
#         positive_indices = (self.targets == anchor_label).nonzero().flatten()
#         positive_indices = positive_indices[positive_indices != anchor_index]
#
#         # Find negative samples with different labels
#         negative_indices = (self.targets != anchor_label).nonzero().flatten()
#
#         # Select the hardest positive sample
#         hardest_positive_index = self.get_hardest_sample(anchor_index, positive_indices)
#
#         # Select hardest negative sample
#         hardest_negative_index = self.get_hardest_sample(anchor_index, negative_indices)
#
#         # Load the images
#         anchor_image_path = self.samples[anchor_index][0]
#         positive_image_path = self.samples[hardest_positive_index][0]
#         negative_image_path = self.samples[hardest_negative_index][0]
#
#         anchor_image = Image.open(anchor_image_path)
#         positive_image = Image.open(positive_image_path)
#         negative_image = Image.open(negative_image_path)
#
#         if self.transform is not None:
#             anchor_image = self.transform(anchor_image)
#             positive_image = self.transform(positive_image)
#             negative_image = self.transform(negative_image)
#
#         return anchor_image, positive_image, negative_image
#
#     # def get_hardest_sample(self, anchor_index, indices):
#     #     anchor_path, _ = self.samples[anchor_index]
#     #     anchor_features = Image.open(anchor_path)
#     #     if self.transform is not None:
#     #         anchor_features = self.transform(anchor_features)
#     #     distances = []
#     #     for index in indices:
#     #         features, _ = self.samples[index]
#     #         features = Image.open(features)
#     #         if self.transform is not None:
#     #             features = self.transform(features)
#     #             distance = torch.norm(anchor_features - features)
#     #             distances.append(distance.item())
#     #
#     #     hardest_index = indices[distances.index(max(distances))]
#     #     return hardest_index
#
#     def get_hardest_sample(self, anchor_index, indices):
#         anchor_path, _ = self.samples[anchor_index]
#         anchor_image = Image.open(anchor_path)
#         if self.transform is not None:
#             anchor_image = self.transform(anchor_image)
#
#         anchor_features = anchor_image  # Use anchor image as anchor features
#
#         # Load and transform images for all indices
#         images = [Image.open(self.samples[index][0]) for index in indices]
#         if self.transform is not None:
#             images = [self.transform(image) for image in images]
#
#         # Convert images to a tensor
#         images = torch.stack(images)
#
#         # Calculate feature distances
#         distances = torch.norm(anchor_features - images, dim=1)
#         mean_distances = torch.mean(distances, dim=(1, 2))
#
#         # Find the index of the maximum mean distance
#         max_distance_index = torch.argmax(mean_distances)
#
#         # Obtain the index of the hardest sample
#         hardest_index = indices[max_distance_index]
#
#         return hardest_index
#
#     def __getitem__(self, index):
#         return self.get_hard_triplet(anchor_index=index)
#
#     def __len__(self):
#         return len(self.samples)


# class SignatureDataset(Dataset):
#     def __init__(self, genuine_folder, forged_folder, transform=None):
#         self.genuine_dataset = CustomDataset(genuine_folder, transform=transform)
#         self.forged_dataset = CustomDataset(forged_folder, transform=transform)
#         self.transform = transform
#
#     def get_hard_triplet(self, anchor_index):
#         anchor_image, positive_image, negative_image = self.genuine_dataset.get_hard_triplet(anchor_index)
#
#         # Find the hardest negative sample from the forged dataset
#         hardest_negative_index = self.forged_dataset.get_hardest_sample(anchor_index)
#
#         # Load the images from the forged dataset
#         negative_image_path = self.forged_dataset.samples[hardest_negative_index][0]
#         negative_image = Image.open(negative_image_path)
#         if self.transform is not None:
#             negative_image = self.transform(negative_image)
#
#         return anchor_image, positive_image, negative_image
#
#     # def get_hardest_sample(self, anchor_index):
#     #     return self.genuine_dataset.get_hardest_sample(anchor_index)
#
#     def __getitem__(self, index):
#         return self.get_hard_triplet(anchor_index=index)
#
#     def __len__(self):
#         return len(self.genuine_dataset)


# class SignatureDataset2(Dataset):
#     def __init__(self, genuine_folder, forged_folder, transform=None):
#         self.genuine_dataset = CustomDataset(genuine_folder, transform=transform)
#         self.forged_dataset = CustomDataset(forged_folder, transform=transform)
#         self.transform = transform
#
#     def get_hard_triplet(self, anchor_index):
#         anchor_image, positive_image, negative_image = self.genuine_dataset.get_hard_triplet(anchor_index)
#
#         # Find the hardest negative sample from the forged dataset
#         # hardest_negative_index = self.get_hardest_sample(anchor_index, self.forged_dataset)
#
#         # Load the images from the forged dataset
#         negative_image_path = self.forged_dataset.samples[anchor_index][0]
#         negative_image = Image.open(negative_image_path)
#         if self.transform is not None:
#             negative_image = self.transform(negative_image)
#
#         return anchor_image, positive_image, negative_image
#
#     def get_hardest_sample(self, anchor_index, dataset):
#         anchor_path, _ = dataset.samples[anchor_index]
#         anchor_image = Image.open(anchor_path)
#         if self.transform is not None:
#             anchor_image = self.transform(anchor_image)
#
#         # Extract features using a pre-trained model
#         model = torchvision.models.resnet50(pretrained=True)
#         # change to accept grayscale
#         model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         model.eval()
#         with torch.no_grad():
#             anchor_features = model(torch.unsqueeze(anchor_image, 0))
#
#             distances = []
#             for index in range(len(dataset)):
#                 if index != anchor_index:
#                     image_path, _ = dataset.samples[index]
#                     image = Image.open(image_path)
#                     if self.transform is not None:
#                         image = self.transform(image)
#                     features = model(torch.unsqueeze(image, 0))
#                     distance = torch.norm(anchor_features - features)
#                     distances.append(distance.item())
#
#         hardest_index = distances.index(max(distances))
#         return hardest_index
#
#     def __getitem__(self, index):
#         return self.get_hard_triplet(anchor_index=index)
#
#     def __len__(self):
#         return len(self.genuine_dataset)


class TripletSignatureDataset(Dataset):
    def __init__(self, genuine_folder, forged_folder, transform=None, test_split=0.2):
        self.genuine_folder = genuine_folder
        self.forged_folder = forged_folder
        self.transform = transform

        self.genuine_dataset = ImageFolder(genuine_folder, transform=transform)
        self.forged_dataset = ImageFolder(forged_folder, transform=transform)

        self.test_split = test_split
        self.train_triplets, self.test_triplets = self.generate_triplets()

    def generate_triplets(self):
        train_triplets = []
        test_triplets = []

        for i in range(len(self.genuine_dataset)):
            genuine_img_path, genuine_class = self.genuine_dataset.imgs[i]
            forged_img_path, _ = self.forged_dataset.imgs[i]

            genuine_img = Image.open(genuine_img_path)
            forged_img = Image.open(forged_img_path)

            if self.transform is not None:
                genuine_img = self.transform(genuine_img)
                forged_img = self.transform(forged_img)

            # Find a positive image with the same class as the genuine image
            positive_idx = random.choice(
                [idx for idx in range(len(self.genuine_dataset)) if self.genuine_dataset.imgs[idx][1] == genuine_class]
            )
            positive_img_path, _ = self.genuine_dataset.imgs[positive_idx]
            positive_img = Image.open(positive_img_path)

            if self.transform is not None:
                positive_img = self.transform(positive_img)

            triplet = (genuine_img, positive_img, forged_img)

            if random.random() < self.test_split:
                test_triplets.append(triplet)
            else:
                train_triplets.append(triplet)

        return train_triplets, test_triplets

    def __len__(self):
        return len(self.train_triplets)

    def __getitem__(self, idx):
        return self.train_triplets[idx]


class PairSignatureDataset(Dataset):
    def __init__(self, genuine_folder, forged_folder, transform=None, test_split=0.2):
        self.genuine_folder = genuine_folder
        self.forged_folder = forged_folder
        self.transform = transform

        self.genuine_dataset = ImageFolder(genuine_folder, transform=transform)
        self.forged_dataset = ImageFolder(forged_folder, transform=transform)

        self.test_split = test_split
        self.train_pairs, self.test_pairs = self.generate_pairs()

    def generate_pairs(self):
        train_pairs = []
        test_pairs = []

        for i in range(len(self.genuine_dataset)):
            genuine_img_path, genuine_class = self.genuine_dataset.imgs[i]
            forged_img_path, _ = self.forged_dataset.imgs[i]

            genuine_img = Image.open(genuine_img_path)
            forged_img = Image.open(forged_img_path)

            if self.transform is not None:
                genuine_img = self.transform(genuine_img)
                forged_img = self.transform(forged_img)

            # Find a positive image with the same class as the genuine image
            positive_idx = random.choice(
                [idx for idx in range(len(self.genuine_dataset)) if self.genuine_dataset.imgs[idx][1] == genuine_class]
            )
            positive_img_path, _ = self.genuine_dataset.imgs[positive_idx]
            positive_img = Image.open(positive_img_path)

            if self.transform is not None:
                positive_img = self.transform(positive_img)

            pair = (genuine_img, forged_img)

            if random.random() < self.test_split:
                test_pairs.append(pair)
            else:
                train_pairs.append(pair)

        return train_pairs, test_pairs

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        return self.train_pairs[idx]


std = 0.0799066424369812
mean = 0.9415700435638428
transformation = transforms.Compose([
    transforms.Resize((110, 225)),
    # transforms.CenterCrop(155),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[mean], std=[std])
])
mean_test = 0.9414753317832947
std_test = 0.07995499670505524

transformation_test = transforms.Compose([
    transforms.Resize((110, 225)),
    # transforms.CenterCrop(155),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean_test], std=[std_test])
])

original_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_org'
forg_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/CEDAR/full_new_forg'

bh_original_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_org'
bh_forg_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_forg'


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(inplace=True),
        nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5))


class SimpleBranch(nn.Module):

    def __init__(self):
        super(SimpleBranch, self).__init__()
        self.nin_block1 = nn.Sequential(
            nin_block(out_channels=96, kernel_size=11, strides=4, padding=0),
            nn.Dropout(p=0.3)
        )
        self.nin_block2 = nn.Sequential(
            nin_block(out_channels=128, kernel_size=2, strides=1, padding=0),
            nn.Dropout(p=0.3)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.nin_block1(x)
        x = self.nin_block2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        out = self.fc(x)
        return out


class SiameseNetwork(nn.Module):
    def __init__(self, branch):
        super(SiameseNetwork, self).__init__()
        self.branch = branch

    def forward(self, anchor_img, positive_img, negative_img):
        anchor_features = self.branch(anchor_img)
        positive_features = self.branch(positive_img)
        negative_features = self.branch(negative_img)

        return anchor_features, positive_features, negative_features


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
            nn.Conv2d(96, 128, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [36, 52]
            nn.Dropout2d(p=0.3),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            # nn.Conv2d(256, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),  # size = [18, 26]
            nn.Dropout2d(p=0.3),
            nn.Flatten(1, -1),  # 18*26*256
            nn.LazyLinear(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
        )

        # TODO: init bias = 0

    def forward(self, x1, x2, x3):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)
        return x1, x2, x3


class SiameseNetworkPreTrained(nn.Module):
    def __init__(self):
        super(SiameseNetworkPreTrained, self).__init__()

        # Load a pre-trained model
        self.pretrained_model = models.alexnet(pretrained=True)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Modify the input channels of the first convolutional layer
        self.pretrained_model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding_mode='zeros')

        # Modify the last fully connected layer for your desired output size
        num_features = self.pretrained_model.classifier[6].in_features
        self.pretrained_model.classifier[6] = nn.Linear(num_features, 32)

        # Add dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def embedding(self, x):
        output = self.pretrained_model(x)
        return output

    def forward(self, anchor_img, positive_img, negative_img):
        anchor_features = self.pretrained_model(anchor_img)
        positive_features = self.pretrained_model(positive_img)
        negative_features = self.pretrained_model(negative_img)

        return anchor_features, positive_features, negative_features


class TripletLoss(nn.Module):
    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean((distance_positive - distance_negative + self.margin).clamp(min=0))
        return loss


dataset = TripletSignatureDataset(genuine_folder=original_signatures_root, forged_folder=forg_signatures_root,
                                  transform=transformation, test_split=0.2)

# train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
# Load the training dataset
batch_size = 32
train_dataset = dataset.train_triplets
test_dataset = dataset.test_triplets
# Create a DataLoader for training set
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create a DataLoader for testing set
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f" train dataset size: {len(train_dataset)}")
print(f" test dataset size: {len(test_dataset)}")
print(f"batch per epoch {len(train_dataset) // batch_size}")
simple_branch = SimpleBranch()
siamese_net = SiameseNetwork(simple_branch)
loss_fn = TripletLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

counter = []
loss_history = []
iteration_number = 0
num_epochs = 20
print_every = 10

# siamese_net.train()
# # # Iterate through the epochs
# for epoch in range(num_epochs):
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
#
#         # Every 10 batches print out the loss
#         if index % print_every == print_every - 1:
#             print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
#             iteration_number += print_every
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
#     scheduler.step(loss_contrastive)
# plt.plot(counter, loss_history)
# plt.show()
#
# torch.save(siamese_net.state_dict(), "triplet_sign1.0.pt")

test_model = SiameseNetwork(simple_branch)
state_dict = torch.load('triplet_sign1.0.pt')
test_model.load_state_dict(state_dict)
test_model.eval()
# #
# test_dataset = TripletSignatureDataset(genuine_folder=bh_original_signatures_root,
#                                        forged_folder=bh_forg_signatures_root,
#                                        transform=transformation_test)
# test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1)
# print(f"test dataset size: {len(test_dataset)}")
#
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)
correct = 0

for i in range(15):
    # Iterate over 5 images and test them with the first image (x0)
    _, x1, x2 = next(dataiter)

    # Concatenate the three images together
    concatenated = torch.cat((x0, x1, x2), 0)

    output1, output2, output3 = test_model(x0, x1, x2)
    # output1 = output1.squeeze()
    # output2 = output2.squeeze()
    # output3 = output3.squeeze()

    # Compute distances
    distance_positive = torch.nn.functional.pairwise_distance(output1, output2)
    distance_negative = torch.nn.functional.pairwise_distance(output1, output3)

    imshow(torchvision.utils.make_grid(concatenated),
           f"Dissimilarity Positive: {distance_positive.item():.2f}, Dissimilarity Negative: {distance_negative.item():.2f}")

if __name__ == '__main__':
    print()
