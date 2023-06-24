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
transformation = transforms.Compose([transforms.Resize((100, 100)),
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
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# class SiameseNet1(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SiameseNet1, self).__init__()
#         self.base_net = models.densenet121(pretrained=True)
#         num_features = self.base_net.classifier.in_features
#         self.head_net = nn.Sequential(nn.LazyLinear(num_features), nn.ReLU(inplace=True),
#                                       nn.LazyLinear(1024), nn.ReLU(inplace=True), nn.LazyLinear(512),
#                                       nn.ReLU(inplace=True), nn.LazyLinear(num_classes))
#
#     def forward_once(self, x):
#         x = x.repeat(1, 3, 1, 1)
#         out = self.base_net(x)
#         out = out.view(out.size()[0], -1)
#         out = self.head_net(out)
#         return out
#
#     def forward(self, x1, x2):
#         out1 = self.forward_once(x1)
#         out2 = self.forward_once(x2)
#         return out1, out2


# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


class SignetY(nn.Module):
    def __init__(self, num_classes=512):
        super(SignetY, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.Dropout(p=0.5),
            nn.Linear(num_classes, num_classes)
        )
        self.resnet = resnet
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=11)
        self.initialize_weights(self.resnet)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 512), 6
        )

    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)

        x1 = self.transformer_encoder(x1)
        x2 = self.transformer_encoder(x2)

        return x1, x2

    def initialize_weights(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 1:  # grayscale input
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


print(f"Using device: {device}")

net = SignetY().to(device)
# Load the training dataset
train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=32)
test_dataloader = DataLoader(siamese_dataset_test, shuffle=True, batch_size=1)
print("dataset size:", len(siamese_dataset))
print("test dataset size", siamese_dataset_test[0])
optimizer = optim.Adam(net.parameters(), lr=1e-5)
# net = SiameseNet1().to(device)
criterion = ContrastiveLoss().to(device)
# optimizer = optim.Adam(net.parameters(), lr=1e-5)

counter = []
loss_history = []
accu_history = []
iteration_number = 0

net.train()

for epoch in range(20):

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

        # Every 20 batches print out the loss
        if i % 20 == 0:
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# plt.plot(loss_history, label="Loss")
# plt.plot(accu_history, label="Test Accuracy")
# plt.show()
# # Iterate through the epochs
# for epoch in range(20):
#
#     # Iterate over batches
#     for i, (img0, img1, label) in enumerate(train_dataloader, 0):
#
#         # Send the images and labels to device
#         img0, img1, label = img0.to(device), img1.to(device), label.to(device)
#
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

show_plot(counter, loss_history)

# evaluate
with torch.no_grad():
    net.eval()
    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(5):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
    # running_loss = 0
    # number_samples = 0
    # distances = []
    # correct = 0
    # total = 0
    # batch_losses = []
    # batch_accs = []
    #
    # for batch_idx, (x1, x2, y) in enumerate(test_dataloader, 0):
    #     x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    #     x1, x2 = net(x1, x2)
    #     loss = criterion(x1, x2, y)
    #     distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))
    #
    #     # track batch loss and accuracy
    #     batch_losses.append(loss.item())
    #     _, predicted = torch.max(torch.pairwise_distance(x1, x2, 2).data, 1)
    #     predicted = predicted == y
    #     batch_accs.append(predicted.sum().item() / len(predicted))
    #     number_samples += len(x1)
    #     running_loss += loss.item() * len(x1)
    #
    #     if (batch_idx + 1) % 20 == 0 or batch_idx == len(test_dataloader) - 1:
    #         print('{}/{}: Loss: {:.4f} | Batch Accuracy: {:.2f}%'.format(batch_idx + 1, len(test_dataloader),
    #                                                                      running_loss / number_samples,
    #                                                                      100 * sum(batch_accs) / len(batch_accs)))
    # distances, y = zip(*distances)
    # distances, y = torch.tensor(distances), torch.tensor(y)
    # max_accuracy = accuracy(distances, y)
    #
    # # calculate and print the overall test loss and accuracy
    # avg_loss = running_loss / number_samples
    # avg_acc = max_accuracy
    # print(f'Test: Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%')
    #
    # # plot the batch losses and accuracies
    # if len(batch_losses) > 1:
    #     fig, ax1 = plt.subplots()
    #     ax1.set_xlabel('Batch')
    #     ax1.set_ylabel('Loss')
    #     ax1.plot(batch_losses, color='tab:red')
    #     ax2 = ax1.twinx()
    #     ax2.set_ylabel('Accuracy')
    #     ax2.plot(batch_accs, color='tab:blue')
    #     fig.tight_layout()
    #     plt.show()
    # else:
    #     print('Batch size is 1. Cannot plot batch losses and accuracies.')
    #     print(f'Loss: {batch_losses[0]:.4f} | Accuracy: {batch_accs[0]:.2f}%')

if __name__ == '__main__':
    print()
