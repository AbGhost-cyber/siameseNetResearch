import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import ImageOps
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import optim
import torch.nn.functional as F


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def accuracy(distances, y, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0
    same_id = (y == 1)

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d + step, step):
        true_positive = (distances <= threshold_d) & same_id
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


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super(SiameseMNIST, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            ImageOps.invert,
            transforms.ToTensor(),
            transforms.Normalize((0.3717,), (0.2831,))
        ])

    def __getitem__(self, index):
        img1, label1 = super(SiameseMNIST, self).__getitem__(index)
        img2, label2 = super(SiameseMNIST, self).__getitem__(torch.randint(len(self), size=(1,)).item())
        label = torch.tensor(int(label1 == label2), dtype=torch.float32)
        return img1, img2, label


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
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

    def forward_once(self, x):
        output = self.shared_conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


# Create instances of the training and test datasets
train_dataset = SiameseMNIST(root='./data', train=True, download=True)
test_dataset = SiameseMNIST(root='./data', train=False, download=True)

# Define the size of the subset, here we wish to train on just 20 percent
subset_size = int(len(train_dataset) * 0.3)

# Define the indices for the subset, generates random perm from 0 to len of train_dataset-1 and selects the
# first subset_size elements from the randomly shuffled indices
subset_indices = torch.randperm(len(train_dataset))[:subset_size]

# Check the size of the subset
print("Subset size:", len(subset_indices))

# Define the subset sampler, Useful in situations where you want to train or validate your model on
# a smaller subset of the dataset, while still preserving the randomness and distribution of the original dataset.
subset_sampler = SubsetRandomSampler(subset_indices)

# Define the data loader using the subset sampler
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define your stuff
siamese_net = SiameseNetwork()
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

counter = []
loss_history = []
iteration_number = 0
print_every = 50

# siamese_net.train()
# # Iterate through the epochs
# for epoch in range(20):
#     running_loss = 0.0
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
#         running_loss += loss_contrastive.item()
#
#         if i % print_every == print_every - 1:
#             print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / print_every:.6f}")
#             iteration_number += print_every
#             counter.append(iteration_number)
#             loss_history.append(running_loss / print_every)
#             running_loss = 0.0
#     print(f"Epoch {epoch + 1} complete")
#
# show_plot(counter, loss_history)
#
# torch.save(siamese_net.state_dict(), "siamese_subset.pt")

# load
test_model = SiameseNetwork()
state_dict = torch.load('siamese_subset.pt')
test_model.load_state_dict(state_dict)
test_model.eval()


@torch.no_grad()
def do_eval(log_interval=50):
    running_loss = 0
    number_samples = 0

    distances = []
    labels = []

    for batch_idx, (x1, x2, y) in enumerate(test_dataloader):

        x1, x2 = test_model(x1, x2)
        loss = loss_fn(x1, x2, y)
        distances.extend(torch.pairwise_distance(x1, x2, 2).cpu().tolist())
        labels.extend(y.tolist())
        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(test_dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(test_dataloader), running_loss / number_samples))

    # distances, y = zip(*distances)
    # distances, y = torch.tensor(distances), torch.tensor(y)
    distances = torch.tensor(distances)
    y = torch.tensor(labels)
    max_accuracy = accuracy(distances, y)
    print(f'Max accuracy: {max_accuracy}')
    return running_loss / number_samples, max_accuracy


losses = []
accuracies = []
for epoch in range(20):
    print('Evaluating', '-' * 20)
    loss, acc = do_eval()
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
