from abc import ABC

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

x_train = np.array(
    [-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939, -2.22112823])

y_train = np.array(
    [-18.56122168, -24.99658931, -24.41907817, -2.688209, -1.54725306, -19.18190097, 1.74117419, 3.97703338,
     -24.80977847])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(99)

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
lr = 1e-1


class MR(nn.Module, ABC):
    def __init__(self, polynomial_degree):
        super().__init__()
        self.polynomial_degree = polynomial_degree + 1
        self.w = torch.tensor(np.zeros(polynomial_degree + 1), requires_grad=True)

    def forward(self, x):
        ret = 0
        for i in range(self.polynomial_degree):
            ret = ret + self.w[i] * x ** i
        return ret


def main(polynomial_degree, number_of_iterations):
    model = MR(polynomial_degree).to(device)

    optimizer = torch.optim.Adam([model.w], lr=lr)

    loss_fn = nn.MSELoss(reduction='mean')

    for epoch in range(number_of_iterations):
        model.train()
        y_predicted = model(x_train_tensor)
        loss = loss_fn(y_train_tensor, y_predicted)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(model.w)

    y_predicted = model(x_train_tensor).detach().numpy()

    plt.scatter(x_train, y_train, color='red', label='Actual')
    plt.scatter(x_train, y_predicted, color='blue', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    title = str(polynomial_degree) + ' Degree Polynomial'
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(1, 1000)
    main(2, 1000)
    main(3, 10000)
    main(4, 10000)
    main(5, 10000)
