from __future__ import division
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def generate_data(n_class=10, n_dim=2, n_data_per_batch=200):
    cluster_center = generate_cluster_center(n_class, n_dim)
    y = numpy.random.randint(n_class, size=n_data_per_batch)
    n_y = numpy.histogram(y, n_class)
    while any(n_y[0] == 0):
        y = numpy.random.randint(n_class, size=n_data_per_batch)
        n_y = numpy.histogram(y, n_class)
    x = sample(cluster_center, y)
    return [x, y]


def generate_cluster_center(n_class, n_dim):
    cluster_center = numpy.random.uniform(-1, 1, size=(n_class, n_dim))
    return cluster_center


def sample(cluster_center, y, var=0.01):
    x = numpy.zeros((y.shape[0], cluster_center.shape[1]))
    for i in range(y.shape[0]):
        x[i, :] = numpy.random.normal(cluster_center[y[i], :], var)
    return x


class L2LNet(nn.Module):
    def __init__(self, n_class=10, n_dim=2):
        super(L2LNet, self).__init__()
        self.fc = nn.Linear(n_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc_w = nn.Linear(16, n_dim)
        self.fc_b = nn.Linear(16, 1)
        self.n_dim = n_dim
        self.n_class = n_class

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        w = self.fc_w(x)
        b = self.fc_b(x)
        return w, b


def l2l_train(model, n_epoch=500):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        batch = generate_data(model.n_class, model.n_dim)
        x, y = Variable(torch.from_numpy(batch[0])).float(), Variable(torch.from_numpy(batch[1]))
        optimizer.zero_grad()
        w, b = model(x)
        M = Variable(torch.zeros(model.n_class, model.n_dim))
        B = Variable(torch.zeros(model.n_class))
        for k in range(model.n_class):
            M[k] = torch.cat((w[:, 0][y == k].view(-1, 1), w[:, 1][y == k].view(-1, 1)), 1).mean(0)
            B[k] = b[y == k].mean()
        pred = torch.mm(x, M.t()) + B.view(1, -1).expand_as(torch.mm(x, M.t()))
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))


def l2l_validate(model, n_epoch=100):
    val_accuracy = []
    for epoch in range(n_epoch):
        batch = generate_data(model.n_class, model.n_dim)
        x, y = Variable(torch.from_numpy(batch[0])).float(), Variable(torch.from_numpy(batch[1]))
        w, b = model(x)
        M = Variable(torch.zeros(model.n_class, model.n_dim))
        B = Variable(torch.zeros(model.n_class))
        for k in range(model.n_class):
            M[k] = torch.cat((w[:, 0][y == k].view(-1, 1), w[:, 1][y == k].view(-1, 1)), 1).mean(0)
            B[k] = b[y == k].mean()
        output = torch.mm(x, M.t()) + B.view(1, -1).expand_as(torch.mm(x, M.t()))
        pred = output.data.max(1)[1]
        accuracy = pred.eq(y.data).cpu().sum() / y.size()[0]
        print('accuracy: %.2f' % accuracy)
        val_accuracy.append(accuracy)
    return numpy.mean(numpy.asarray(val_accuracy))


def main():
    n_class, n_dim = 20, 2
    model = L2LNet(n_class, n_dim)
    l2l_train(model)
    val_accuracy = l2l_validate(model)
    print('average testing accuracy: %.2f' % val_accuracy)

if __name__ == '__main__':
    main()
