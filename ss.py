import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.models as models
import torch.nn.functional as F


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ResNet18Gray(nn.Module):
    def __init__(self):
        super(ResNet18Gray, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data[:, :1, :, :]
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SiameseNet(nn.Module):
    def __init__(self, resnet):
        super(SiameseNet, self).__init__()
        self.resnet = resnet
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward_once(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        output = self.fc2(torch.abs(output1 - output2))
        output = torch.squeeze(output, dim=1)
        return output


class SiameseMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super(SiameseMNIST, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(100),
            ImageOps.invert,
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        img1, label1 = super(SiameseMNIST, self).__getitem__(index)
        img2, label2 = super(SiameseMNIST, self).__getitem__(torch.randint(len(self), size=(1,)).item())
        label = torch.tensor(int(label1 == label2), dtype=torch.float32)
        return img1, img2, label


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


train_dataset = SiameseMNIST(root='./data', train=True, download=True)
# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, and also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)


# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy().reshape(-1))

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


# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")

net = TestSign().to(device)
# Define your margin schedule
# min_margin = 3
# max_margin = 1
# num_epochs = 50

# Load the training dataset
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
print(len(train_dataset))
optimizer = optim.RMSprop(net.parameters(), lr=1e-5, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
num_epochs = 20

criterion = ContrastiveLoss().to(device)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# margin_schedule = [min_margin + (max_margin - min_margin) * (epoch / (num_epochs - 1)) for epoch in range(num_epochs)]

counter = []
loss_history = []
iteration_number = 0
print_every = 9370

for epoch in range(num_epochs):
    running_loss = 0.0
    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, epoch * len(train_dataloader)):

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
        scheduler.step()

        # Add the loss to the running total
        running_loss += loss_contrastive.item()

        # # Adjust margin based on epoch
        # criterion.margin = margin_schedule[epoch]

        if i % print_every == print_every - 1:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / print_every:.6f}")
            iteration_number += print_every

            counter.append(iteration_number)
            loss_history.append(running_loss / print_every)
            running_loss = 0.0
    print(f"Epoch {epoch + 1} complete")
plt.plot(loss_history, label="Loss")
plt.show()

# train_dataset = SiameseMNIST(root='./data', train=True, download=True)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# print("dataset length: ", len(train_dataset))
# resnet = ResNet18Gray()
# model = SiameseNet(resnet=resnet)
# # xx = torch.randn((1, 1, 224, 224))
# # res = model(xx, xx)
# # print(res.shape)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 10
# loss_history = []
#
# for epoch in range(num_epochs):
#     epoch_loss = 0.0
#     for i, (inputs1, inputs2, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(inputs1, inputs2)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     loss_history.append(epoch_loss / len(train_loader))
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss / len(train_loader)))
#
# plt.plot(range(num_epochs), loss_history)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

if __name__ == '__main__':
    print()
