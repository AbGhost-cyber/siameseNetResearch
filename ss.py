import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps
from matplotlib import pyplot as plt
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import f1_score
import random
from imblearn.over_sampling import ADASYN


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
            transforms.Resize((145, 210)),
            ImageOps.invert,
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


# print(example_batch[2].numpy().reshape(-1))
# imshow(torchvision.utils.make_grid(concatenated))


# Compute the ratio of same-class pairs to different-class pairs in the dataset
# same_class_labels = [label for _, _, label in train_dataset if label == 1]
# diff_class_labels = [label for _, _, label in train_dataset if label == 0]
#
# num_same_class = len(same_class_labels)
# num_diff_class = len(diff_class_labels)
#
# if num_same_class == 0 or num_diff_class == 0:
#     ratio = float('inf') if num_same_class > 0 else 0.0
# else:
#     ratio = num_same_class / num_diff_class
#
# print("Ratio of same-class pairs to different-class pairs:", ratio)
# print("Number of same-class pairs:", num_same_class)
# print("Number of different-class pairs:", num_diff_class)


# Ratio of same-class pairs to different-class pairs: 0.11259457053849577
# 6072
# 53928


class TestSign(nn.Module):

    def __init__(self):
        super(TestSign, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=128, out_channels=384, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.5),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=384 * 7 * 7, out_features=512, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=128, out_features=2, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class TestSign2(nn.Module):

    def __init__(self):
        super(TestSign2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(256, 384, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(384, 256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.2),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

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

net = TestSign2().to(device)

# subset_indices = range(100)  # Specify the number of samples in the subset
# subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
#
# loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1)
# mean = torch.zeros(1)
# std = torch.zeros(1)
#
# # Calculate the mean and standard deviation
# for data1, data2, _ in loader:
#     mean += torch.cat((data1, data2)).mean(dim=(0, 2, 3))
#     std += torch.cat((data1, data2)).std(dim=(0, 2, 3))
#
# mean /= len(subset_dataset) * 2
# std /= len(subset_dataset) * 2

# Mean: tensor([0.3717])
# Standard Deviation: tensor([0.2831])

# Load the training dataset
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
print(len(train_dataset))
optimizer = optim.Adam(net.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
num_epochs = 40

criterion = ContrastiveLoss().to(device)

# counter = []
# loss_history = []
# iteration_number = 0
# print_every = 100
# net.train()
#
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     # Iterate over batches
#     for i, (img0, img1, label) in enumerate(train_dataloader, epoch * len(train_dataloader)):
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
#         # scheduler.step()
#
#         # Add the loss to the running total
#         running_loss += loss_contrastive.item()
#
#         # # Adjust margin based on epoch
#         # criterion.margin = margin_schedule[epoch]
#
#         if i % print_every == print_every - 1:
#             print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / print_every:.6f}")
#             iteration_number += print_every
#
#             counter.append(iteration_number)
#             loss_history.append(running_loss / print_every)
#             running_loss = 0.0
#     print(f"Epoch {epoch + 1} complete")
# plt.plot(loss_history, label="Loss")
# plt.show()
#
# torch.save(net.state_dict(), 'testSignalss1.pt')
# print("saved")

test_dataset = SiameseMNIST(root='./data', train=False, download=True)
transform1 = transforms.Compose([
    transforms.Resize((145, 210)),
    transforms.ToTensor()
])
test_dataset.transform = transform1
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)


@torch.no_grad()
def do_eval(model, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(test_dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(test_dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(test_dataloader), running_loss / number_samples))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max accuracy: {max_accuracy}')
    return running_loss / number_samples, max_accuracy


model = TestSign2().to(device)
state_dict = torch.load('/Users/mac/Downloads/myNewModel.pt', map_location=device)
model.load_state_dict(state_dict)
losses = []
accuracies = []
for epoch in range(20):
    print('Evaluating', '-' * 20)
    loss, acc = do_eval(model)
    losses.append(loss)
    accuracies.append(acc)
# Plotting loss
plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# Plotting accuracy
plt.plot(range(len(accuracies)), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.show()
if __name__ == '__main__':
    print()
