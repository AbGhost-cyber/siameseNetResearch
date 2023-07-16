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
from torchvision.models import densenet121


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


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 96, kernel_size=11, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, stride=2),
        #
        #     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Conv2d(256, 384, 3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Conv2d(384, 512, 3, stride=1, padding=1, padding_mode='zeros'),
        #     nn.ReLU(inplace=True),
        #
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(512 * 4 * 4, 1024),
        #     nn.LeakyReLU(inplace=True),
        #
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(inplace=True),
        #
        #     nn.Linear(512, 2),
        # )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            # nn.Dropout(p=0.2),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 2056),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),

            nn.Linear(2056, 1024),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# Load the training dataset
folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/training/")

# Resize the images and transform to tensors
transformation_train = transforms.Compose([transforms.Resize((100, 100)),
                                           transforms.CenterCrop(100),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.2062], std=[0.1148]),
                                           ])
transformation_test = transforms.Compose([transforms.Resize((100, 100)),
                                          transforms.ToTensor()
                                          ])
# Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation_train)

siamese_net = SiameseNetwork()
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0005)
# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=64)

counter = []
loss_history = []
iteration_number = 0


# siamese_net.train()
# # Iterate through the epochs
# for epoch in range(100):
#
#     # Iterate over batches
#     for i, (img0, img1, label) in enumerate(train_dataloader, 0):
#
#         # Zero the gradients
#         optimizer.zero_grad()
#
#         # Pass in the two images into the network and obtain two outputs
#         output1, output2 = siamese_net(img0, img1)
#
#         # Pass the outputs of the networks and label into the loss function
#         loss_contrastive = loss_fn(output1, output2, label)
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
#
# torch.save(siamese_net.state_dict(), "small_siamese_more_neurons+transforms.pt")
#
#
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# load
test_model = SiameseNetwork()
state_dict = torch.load('small_siamese_more_neurons+transforms.pt')
test_model.load_state_dict(state_dict)
test_model.eval()

# Locate the test dataset and load it into the SiameseNetworkDataset
folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
siamese_dataset1 = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                         transform=transformation_test)
test_dataloader = DataLoader(siamese_dataset1, batch_size=15, shuffle=True)

# Grab one image that we are going to test
# dataiter = iter(test_dataloader)
# x0, _, _ = next(dataiter)
#
# for i in range(15):
#     # Iterate over 5 images and test them with the first image (x0)
#     _, x1, label2 = next(dataiter)
#
#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)
#
#     output1, output2 = test_model(x0, x1)
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
# Test the Siamese neural network
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        inputs1, inputs2, labels = data
        outputs1, outputs2 = test_model(inputs1, inputs2)
        _, predicted = torch.max(outputs1 - outputs2, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test set: %d %%' % (100 * correct / total))
if __name__ == '__main__':
    print()
