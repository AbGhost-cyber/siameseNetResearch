import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define the custom attention module
class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()

        self.attention = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.Softmax(dim=1)
        )

        # self.attention = nn.Linear(input_size, hidden_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.attention(x)


# Define the model
class AttentionModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(AttentionModel, self).__init__()
        self.attention_module = AttentionModule(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        attended_representation = self.attention_module(x)
        output = self.fc(attended_representation)
        return output


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 64
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = AttentionModel(input_size, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_loader)
loss_values = []  # To store the loss values
accuracy_values = []  # To store the accuracy values
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            loss_values.append(loss.item())  # Store the loss value

    # Calculate accuracy after each epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_values.append(accuracy)  # Store the accuracy value

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

# Plot the loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy_values) + 1), accuracy_values)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on Test Set')

plt.tight_layout()
plt.show()

if __name__ == '__main__':
    print()
