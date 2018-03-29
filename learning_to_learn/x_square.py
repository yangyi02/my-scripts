from __future__ import division
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, ndim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(ndim, 16)
        self.fc_1 = nn.Linear(16, 16)
        self.fc_2 = nn.Linear(16, 16)
        self.fc_3 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        y = self.fc2(x)
        return y


def main():
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(2000):
        x = Variable(torch.rand(100,2)) * 10
        y = x * x
        optimizer.zero_grad()
        pred = model(x)
        loss = (pred - y).pow(2).sum()
        loss.backward()
        optimizer.step()

        x = Variable(torch.rand(100, 2)) * 10
        y = x * x
        pred = model(x)
        loss = (pred - y).pow(2).sum()
        print(loss.data.numpy()[0])

if __name__ == '__main__':
    main()