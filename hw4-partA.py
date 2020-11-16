# /**
#  * @author Mia Mohammad Imran
#  */


import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import transforms
from torch import optim
import torchvision.datasets as dsets
import torch.nn as nn

batch_size = 64
hidden_sizes = [32, 16]
output_size = 10
learning_rate = 1e-3

train_dataset = dsets.MNIST(root='torch/MNIST_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='torch/MNIST_data', train=False, transform=transforms.ToTensor(), download=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, hidden_sizes[0], kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_sizes[0], hidden_sizes[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 16, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

time0 = time()
epochs = 15
losses = []
iteration = []
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch {} - Training loss: {}".format(epoch, running_loss / len(trainloader)))
    losses.append(running_loss)
    iteration.append(epoch)

print("Training Time (in minutes) =", (time() - time0) / 60)

plt.plot(iteration, losses, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Loss rate vs. iterations of training dataset (pytorch)')
plt.show()

correct_count, all_count = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_count += labels.size(0)
        correct_count += (predicted == labels).sum()
print("Model Accuracy = ", (correct_count / all_count))