from abc import ABC

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

batch_size = 100
c0 = 1
c1 = 7


def preprocess(dataset):
    dataset = [_ for i, _ in enumerate(dataset) if dataset[i][1] == c0 or dataset[i][1] == c1]
    modified_dataset = []
    for item in dataset:
        temp = list(item)
        if temp[1] == c0:
            temp[1] = -1
        else:
            temp[1] = 1
        modified_dataset.append(tuple(temp))
    return modified_dataset


train_dataset = dsets.MNIST(root='', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='', train=False, transform=transforms.ToTensor(), download=True)

train_dataset = preprocess(train_dataset)
test_dataset = preprocess(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class LogisticRegression(torch.nn.Module, ABC):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


input_dim = 784
output_dim = 2
epochs = 30
lr_rate = 1e-3
model = LogisticRegression(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

iteration = 0

x_iteration_test_error = []
y_error_test = []


def test_error():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, input_dim))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        idx = 0
        for label in predicted:
            if label.item() == 0:
                predicted[idx] = torch.tensor(-1)
            idx = idx + 1

        correct += (predicted == labels).sum()
    accuracy = 100 * torch.true_divide(correct, total)
    x_iteration_test_error.append(iteration)
    y_error_test.append(100.0 - accuracy.item())


x_iteration_train_error = []
y_error_train = []


def train_error():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images.view(-1, input_dim))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        idx = 0
        for label in predicted:
            if label.item() == 0:
                predicted[idx] = torch.tensor(-1)
            idx = idx + 1

        correct += (predicted == labels).sum()
    accuracy = 100 * torch.true_divide(correct, total)
    x_iteration_train_error.append(iteration)
    y_error_train.append(100.0 - accuracy.item())


x_iteration_risk_test = []
y_risk_test = []


def test_risk():
    for images, labels in test_loader:
        images = Variable(images.view(-1, input_dim))
        labels = Variable(labels)
        outputs = model(images)

        idx = 0
        for label in labels:
            if label.item() == -1:
                labels[idx] = torch.tensor(0)
            idx = idx + 1

        loss = criterion(outputs, labels)
        x_iteration_risk_test.append(iteration)
        y_risk_test.append(loss.item())


test_error()
train_error()

x_iteration_risk_train = []
y_risk_train = []

for epoch in range(epochs):
    print(epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, input_dim))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)

        idx = 0
        for label in labels:
            if label.item() == -1:
                labels[idx] = torch.tensor(0)
            idx = idx + 1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iteration = iteration + 1

        x_iteration_risk_train.append(iteration)
        y_risk_train.append(loss.item())
        test_error()
        train_error()
        if iteration < 100 or iteration % 100 == 0:
            test_risk()

print(iteration)
print(y_risk_train)
print(len(y_risk_train))

plt.scatter(x_iteration_risk_train, y_risk_train, color='red', label='Train')
plt.xlabel('x')
plt.ylabel('y')
title = 'Risk value vs. iterations of training dataset'
plt.title(title)
plt.legend()
plt.show()

print(y_risk_test)
print(len(y_risk_test))
# print(y_risk_test)
plt.scatter(x_iteration_risk_test, y_risk_test, color='red', label='Test')
plt.xlabel('x')
plt.ylabel('y')
title = 'Risk value vs. iterations of test dataset'
plt.title(title)
plt.legend()
plt.show()

print(y_error_train)
print(y_error_test)

plt.plot(x_iteration_train_error, y_error_train, color='blue', label='Train')
plt.plot(x_iteration_test_error, y_error_test, color='red', label='Test')
plt.xlabel('x')
plt.ylabel('y')
title = 'Error vs. iterations of dataset'
plt.title(title)
plt.legend()
plt.show()
