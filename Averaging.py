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
import torchvision.models as models
from torchsummary import summary


# The passage (Model soups: averaging weights of multiple fine-tuned models improves accuracy without
# increasing inference time) discusses a paper that reexamines the conventional approach to maximizing model accuracy
# in the context of fine-tuning large pre-trained models. The typical approach involves training multiple models
# with different hyperparameter configurations and selecting the best-performing model based on a validation set.
# However, the paper suggests that instead of discarding the other models,
# averaging their weights can lead to improved accuracy and robustness.

# In the case of Siamese networks, it might not be straightforward to apply the averaging of model weights because
# the network's architecture and objective function are designed specifically for pair-wise comparisons.
# Averaging the weights of multiple Siamese networks could potentially disrupt the learned similarity metric
# and result in decreased performance.
# However, it's worth noting that there are ensemble techniques specifically designed for Siamese networks.
# One common approach is to create an ensemble by training multiple Siamese networks independently and then
# combining their predictions using techniques like voting or averaging at the similarity or distance level.
# So, while the specific approach of averaging model weights might not be directly applicable to Siamese networks,
# ensemble techniques tailored for these networks can still be effective for improving performance.

def calculate_FAR_FRR(threshold, model, dataloader):
    # FAR = (False Positive / (False Positive + True Negative)) * 100
    # FRR = (False Negative / (False Negative + True Positive)) * 100

    far_count = 0  # Counter for false acceptance
    frr_count = 0  # Counter for false rejection
    impostor_count = 0  # Counter for impostor pairs
    genuine_count = 0  # Counter for genuine pairs

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            image1, image2, label = batch

            # Pass samples through the model to get embeddings
            first_emb, second_emb = model(image1, image2)

            # Compute distances
            distances = F.pairwise_distance(first_emb, second_emb, p=2)

            # Convert distances to binary predictions
            predictions = (distances <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise

            # Update counts
            for i in range(len(label)):
                if label[i] == 1:  # Genuine pair
                    genuine_count += 1
                    if predictions[i] == 0:  # Incorrectly rejected
                        frr_count += 1
                else:  # Impostor pair
                    impostor_count += 1
                    if predictions[i] == 1:  # Incorrectly accepted
                        far_count += 1

    far = (far_count / impostor_count) * 100
    frr = (frr_count / genuine_count) * 100

    return far, frr


def calculate_accuracy(threshold, model, dataloader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            image1, image2, label = batch

            # Pass samples through the model to get embeddings
            first_emb, second_emb = model(image1, image2)

            # Compute distances
            distances = F.pairwise_distance(first_emb, second_emb, p=2)

            # Convert distances to binary predictions
            predictions = (distances <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise

            # Update correct and total counts
            # ground truth labels are in the third element of the batch tuple
            correct += torch.eq(predictions, label).all(dim=1).sum().item()
            total += len(image1)

    accuracy = correct / total

    return accuracy * 100


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

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, 32)
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
        self.dropout = nn.Dropout(p=0.4)

    def embedding(self, x):
        output = self.pretrained_model(x)
        return output

    def forward(self, input1, input2):
        output1 = self.embedding(input1)
        output2 = self.embedding(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
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
mean = 0.2062
std = 0.1148
normalize = transforms.Normalize(mean=[mean], std=[std])
transformation_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.CenterCrop(150),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    normalize
])
transformation_test = transforms.Compose([transforms.Resize((150, 150)),
                                          transforms.ToTensor(),
                                          normalize
                                          ])

# # Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation_train)

siamese_net = SiameseNetworkPreTrained()
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.00005)
# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=64)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
print(f"dataset size: {len(siamese_dataset)}")
print(f"batch per epoch {len(siamese_dataset) // 64}")
counter = []
loss_history = []
iteration_number = 0
epoch = 30
# Iterate through the epochs
# for epoch in range(epoch):
#     siamese_net.train()
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
#         # scheduler.step(loss_contrastive)
#
#         # Every 10 batches print out the loss
#         if i % 10 == 0:
#             print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
#             iteration_number += 10
#
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
# show_plot(counter, loss_history)
#
# torch.save(siamese_net.state_dict(), "small_siamese_dropout11.pt")


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
test_model = SiameseNetworkPreTrained()
state_dict = torch.load('small_siamese_dropout11.pt')
test_model.load_state_dict(state_dict)
test_model.eval()

# Locate the test dataset and load it into the SiameseNetworkDataset
folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
siamese_dataset1 = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                         transform=transformation_test)
test_dataloader = DataLoader(siamese_dataset1, batch_size=1, shuffle=True)

model_accuracy = calculate_accuracy(threshold=1, model=test_model, dataloader=test_dataloader)
model_far, model_frr = calculate_FAR_FRR(threshold=1, model=test_model, dataloader=test_dataloader)
print(f"model accuracy: {model_accuracy}")
print(f"model false rejection rate: {model_frr}")
print(f"model false acceptance rate: {model_far}")

# Grab one image that we are going to test
# dataiter = iter(test_dataloader)
# example_batch = next(iter(test_dataloader))
# concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
# print(example_batch[2].numpy().reshape(-1))
#
# imshow(torchvision.utils.make_grid(concatenated))
# x0, _, label1 = next(dataiter)
# correct = 0
# for i in range(15):
#     # Iterate over 5 images and test them with the first image (x0)
#     _, x1, label2 = next(dataiter)
#
#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)
#
#     output1, output2 = test_model(x0, x1)
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     print(label1, label2)
#     print(torch.eq(label1, label2))
#     imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
if __name__ == '__main__':
    print()
