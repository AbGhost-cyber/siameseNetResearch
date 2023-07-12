import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import ImageOps


# Define the Siamese neural network architecture
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


# Define the SiameseMNIST dataset class
class SiameseMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super(SiameseMNIST, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(50),
            ImageOps.invert,
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.3717,), (0.2831,))
        ])

    def __getitem__(self, index):
        img1, label1 = super(SiameseMNIST, self).__getitem__(index)
        img2, label2 = super(SiameseMNIST, self).__getitem__(torch.randint(len(self), size=(1,)).item())
        label = torch.tensor(int(label1 == label2), dtype=torch.float32)
        return img1, img2, label


# Create instances of the training and test datasets
train_dataset = SiameseMNIST(root='./data', train=True, download=True)
test_dataset = SiameseMNIST(root='./data', train=False, download=True)

# Create data loaders for the training and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Create an instance of the Siamese neural network and define the loss function and optimizer
model = SiameseNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Siamese neural network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs1, inputs2, labels = data
        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs1, inputs2)
        loss = criterion(outputs1 - outputs2, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Test the Siamese neural network
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs1, inputs2, labels = data
        outputs1, outputs2 = model(inputs1, inputs2)
        _, predicted = torch.max(outputs1 - outputs2, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    print()
