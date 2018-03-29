from __future__ import division
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def generate_data(cluster_center, n_data_per_batch=100):
    (n_class, n_dim) = cluster_center.shape
    y = numpy.random.randint(n_class, size=n_data_per_batch)
    x = sample(cluster_center, y)
    return [x, y]


def generate_cluster_center(n_class, n_dim):
    cluster_center = numpy.random.uniform(-1, 1, size=(n_class, n_dim))
    if n_dim == 1:
        cluster_center = numpy.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
    return cluster_center


def sample(cluster_center, y, var=0.1):
    x = numpy.zeros((y.shape[0], cluster_center.shape[1]))
    for i in range(y.shape[0]):
        x[i, :] = numpy.random.normal(cluster_center[y[i], :], var)
    return x


class Net(nn.Module):
    def __init__(self, n_class=10, n_dim=2):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_dim, n_class)

    def forward(self, x):
        y = self.fc(x)
        return y


def train(model, cluster_center, n_epoch=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        batch = generate_data(cluster_center)
        x, y = Variable(torch.from_numpy(batch[0])).float(), Variable(torch.from_numpy(batch[1]))
        optimizer.zero_grad()
        pred = model(x)
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))


def validate(model, cluster_center, n_epoch=1000):
    val_accuracy = []
    for epoch in range(n_epoch):
        batch = generate_data(cluster_center)
        x, y = Variable(torch.from_numpy(batch[0])).float(), Variable(torch.from_numpy(batch[1]))
        output = model(x)
        pred = output.data.max(1)[1]
        accuracy = pred.eq(y.data).cpu().sum() * 1.0 / y.size()[0]
        print('accuracy: %.2f' % accuracy)
        val_accuracy.append(accuracy)
    return numpy.mean(numpy.asarray(val_accuracy))


def main():
    n_class, n_dim = 20, 2
    cluster_center = generate_cluster_center(n_class, n_dim)
    model = Net(n_class, n_dim)
    train(model, cluster_center)
    val_accuracy = validate(model, cluster_center)
    print('average accuracy: %.2f' % val_accuracy)
    if n_dim == 1:
        print(cluster_center)
        for param in model.parameters():
            print(param.data)

if __name__ == '__main__':
    main()
