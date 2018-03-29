from __future__ import division
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


def generate_data_l(cluster_center, n_data_per_batch=75):
    (n_class, n_dim) = cluster_center.shape
    y = numpy.random.randint(n_class, size=n_data_per_batch)
    n_y = numpy.histogram(y, n_class)
    while any(n_y[0] == 0):
        y = numpy.random.randint(n_class, size=n_data_per_batch)
        n_y = numpy.histogram(y, n_class)
    x = sample(cluster_center, y)
    data = [x, y]
    return data


def generate_data_n(cluster_center, n_class_n=5, n_data_per_batch=25):
    (n_class_l, n_dim) = cluster_center.shape
    cluster_center_n = generate_cluster_center(n_class_n, n_dim)
    y = numpy.random.randint(n_class_l, n_class_l + n_class_n, size=n_data_per_batch)
    n_y = numpy.histogram(y, n_class_n)
    while any(n_y[0] == 0):
        y = numpy.random.randint(n_class_l, n_class_l + n_class_n, size=n_data_per_batch)
        n_y = numpy.histogram(y, n_class_n)
    cluster_center = numpy.concatenate((cluster_center, cluster_center_n))
    x = sample(cluster_center, y)
    data = [x, y]
    return data


def generate_cluster_center(n_class=15, n_dim=2):
    cluster_center = numpy.random.uniform(-1, 1, size=(n_class, n_dim))
    return cluster_center


def sample(cluster_center, y, var=0.1):
    x = numpy.zeros((y.shape[0], cluster_center.shape[1]))
    for i in range(y.shape[0]):
        x[i, :] = numpy.random.normal(cluster_center[y[i], :], var)
    return x


class Net(nn.Module):
    def __init__(self, n_dim=2, n_class=15):
        super(Net, self).__init__()
        self.fc_w = nn.Linear(n_dim, n_dim)
        self.fc_b = nn.Linear(n_dim, n_dim)
        self.n_dim = n_dim
        self.n_class = n_class

    def forward(self, x):
        w = self.fc_w(x)
        b = self.fc_b(x)
        b = - torch.sum(b * b, 1)
        return w, b


class L2LNet(nn.Module):
    def __init__(self, n_dim=2, n_class_lifelong=15, n_class_new=5):
        super(L2LNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_w = nn.Linear(32, n_dim)
        self.fc_b = nn.Linear(32, 1)
        self.n_dim = n_dim
        self.n_class_n = n_class_new
        self.n_class_l = n_class_lifelong

        self.fc_l = nn.Linear(n_dim, n_class_lifelong)

    def forward(self, x_lifelong, x_new):
        x = F.relu(self.fc1(x_new))
        x = F.relu(self.fc2(x))
        w = self.fc_w(x)
        b = self.fc_b(x * x)

        y_l = self.fc_l(x_lifelong)
        y_n = self.fc_l(x_new)
        return y_l, y_n, w, b


def l2l_train(model, cluster_center, n_epoch=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        data_l = generate_data_l(cluster_center)
        data_n = generate_data_n(cluster_center, model.n_class_n)
        optimizer.zero_grad()
        x_l, y_l = Variable(torch.from_numpy(data_l[0])).float(), Variable(
            torch.from_numpy(data_l[1]))
        x_n, y_n = Variable(torch.from_numpy(data_n[0])).float(), Variable(
            torch.from_numpy(data_n[1]))
        pred_ll, pred_nl, w, b = model(x_l, x_n)
        M = Variable(torch.zeros(model.n_class_n, model.n_dim))
        B = Variable(torch.zeros(model.n_class_n))
        for k in range(model.n_class_n):
            M[k] = torch.cat((w[:, 0][y_n == model.n_class_l + k].view(-1, 1),
                              w[:, 1][y_n == model.n_class_l + k].view(-1, 1)), 1).mean(0)
            B[k] = b[y_n == model.n_class_l + k].mean()
        pred_ln = torch.mm(x_l, M.t()) + B.view(1, -1).expand_as(torch.mm(x_l, M.t()))
        pred_nn = torch.mm(x_n, M.t()) + B.view(1, -1).expand_as(torch.mm(x_n, M.t()))
        pred = torch.cat((torch.cat((pred_ll, pred_nl)), torch.cat((pred_ln, pred_nn))), 1)
        y = torch.cat((y_l, y_n))
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))


def l2l_validate(model, cluster_center, n_epoch=100):
    val_accuracy = []
    for epoch in range(n_epoch):
        data_l = generate_data_l(cluster_center)
        data_n = generate_data_n(cluster_center, model.n_class_n)
        x_l, y_l = Variable(torch.from_numpy(data_l[0])).float(), Variable(
            torch.from_numpy(data_l[1]))
        x_n, y_n = Variable(torch.from_numpy(data_n[0])).float(), Variable(
            torch.from_numpy(data_n[1]))
        pred_ll, pred_nl, w, b = model(x_l, x_n)
        M = Variable(torch.zeros(model.n_class_n, model.n_dim))
        B = Variable(torch.zeros(model.n_class_n))
        for k in range(model.n_class_n):
            M[k] = torch.cat((w[:, 0][y_n == model.n_class_l + k].view(-1, 1),
                              w[:, 1][y_n == model.n_class_l + k].view(-1, 1)), 1).mean(0)
            B[k] = b[y_n == model.n_class_l + k].mean()
        pred_ln = torch.mm(x_l, M.t()) + B.view(1, -1).expand_as(torch.mm(x_l, M.t()))
        pred_nn = torch.mm(x_n, M.t()) + B.view(1, -1).expand_as(torch.mm(x_n, M.t()))
        pred = torch.cat((torch.cat((pred_ll, pred_nl)), torch.cat((pred_ln, pred_nn))), 1)
        pred = pred.data.max(1)[1]
        y = torch.cat((y_l, y_n))
        accuracy = pred.eq(y.data).cpu().sum() * 1.0 / y.size()[0]
        # print('accuracy: %.2f' % accuracy)
        val_accuracy.append(accuracy)
        acc_l = pred.eq(y.data).cpu()[0:75].sum() * 1.0 / 75
        acc_n = pred.eq(y.data).cpu()[75:100].sum() * 1.0 / 25
        print('accuracy: %.2f, lifelong accuracy: %.2f, new accuracy: %.2f' % (accuracy, acc_l, acc_n))

    return numpy.mean(numpy.asarray(val_accuracy))


def main():
    cluster_center = generate_cluster_center()
    model = L2LNet()
    # model = Net()
    l2l_train(model, cluster_center)
    val_accuracy = l2l_validate(model, cluster_center)
    print('average accuracy: %.2f' % val_accuracy)

if __name__ == '__main__':
    main()
