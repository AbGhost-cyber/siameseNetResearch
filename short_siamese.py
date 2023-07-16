import random

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.shared_conv(x)
        output = output.view(output.size(0), -1)
        # output = self.fc(output)
        return output


class SiameseMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super(SiameseMNIST, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(30),
            transforms.CenterCrop(30),
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.3717,), (0.2831,))
        ])

    def __getitem__(self, index):
        img1, label1 = super(SiameseMNIST, self).__getitem__(index)

        # Select a random index for the second image
        if random.random() < 0.5:
            # Select a random index for the same class
            same_class_indices = torch.nonzero(torch.as_tensor(self.targets) == label1).flatten()
            random_index = same_class_indices[random.randint(0, len(same_class_indices) - 1)].item()
        else:
            # Select a random index for a different class
            different_class_indices = torch.nonzero(torch.as_tensor(self.targets) != label1).flatten()
            random_index = different_class_indices[random.randint(0, len(different_class_indices) - 1)].item()

        img2, label2 = super(SiameseMNIST, self).__getitem__(random_index)
        label = torch.tensor(int(label1 == label2), dtype=torch.float32)

        return img1, img2, label


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv_net = ConvNet()

    def forward(self, input_1, input_2):
        output_1 = self.conv_net(input_1)
        output_2 = self.conv_net(input_2)
        distance = torch.abs(output_1 - output_2)
        distance = self.conv_net.fc(distance)
        return distance


siamese_network = SiameseNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)

# Define the training dataset and dataloader
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

train_dataset = SiameseMNIST(root='./data', train=True, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 50
counter = []
loss_history = []
iteration_number = 0
print_every = 50

for epoch in range(20):
    running_loss = 0.0
    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2 = siamese_network(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        running_loss += loss_contrastive.item()

        if i % print_every == print_every - 1:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / print_every:.6f}")
            iteration_number += print_every
            counter.append(iteration_number)
            loss_history.append(running_loss / print_every)
            running_loss = 0.0
    print(f"Epoch {epoch + 1} complete")

plt.plot(counter, loss_history)
plt.show()

torch.save(siamese_network.state_dict(), "siamese_short.pt")

if __name__ == '__main__':
    print()
