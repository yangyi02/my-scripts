import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def run(N=100, D_in=10, D_noise=100, H=1000):
    x1 = Variable(torch.randn(N, D_in), requires_grad=False)
    x2 = Variable(torch.randn(N, D_in), requires_grad=False)
    x3 = Variable(torch.randn(N, D_noise), requires_grad=False)
    w1 = Variable(torch.randn(D_in, 1), requires_grad=False)
    w2 = Variable(torch.randn(D_in, 1), requires_grad=False)
    y = torch.mm(x1, w1) + torch.mm(x2, w2)
    y = (y > 0).float()

    model1 = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, 1),
        torch.nn.Sigmoid(),
    )

    model2 = torch.nn.Sequential(
        torch.nn.Linear(D_in * 2, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, 1),
        torch.nn.Sigmoid(),
    )

    model3 = torch.nn.Sequential(
        torch.nn.Linear(D_in * 2 + D_noise, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, 1),
        torch.nn.Sigmoid(),
    )

    N_test = 10000
    x1_test = Variable(torch.randn(N_test, D_in))
    x2_test = Variable(torch.randn(N_test, D_in))
    x3_test = Variable(torch.randn(N_test, D_noise))
    y_test = torch.mm(x1_test, w1) + torch.mm(x2_test, w2)
    y_test = (y_test > 0).float()

    loss1, test_loss1, acc1, test_acc1 = optimize_model(model1, x1, y, x1_test, y_test)
    loss2, test_loss2, acc2, test_acc2 = optimize_model(model2, torch.cat((x1, x2), 1), y,
                                                        torch.cat((x1_test, x2_test), 1), y_test)
    loss3, test_loss3, acc3, test_acc3 = optimize_model(model3, torch.cat((x1, x2, x3), 1), y,
                                                        torch.cat((x1_test, x2_test, x3_test), 1), y_test)

    return loss1, loss2, loss3, test_loss1, test_loss2, test_loss3, acc1, acc2, acc3, test_acc1, test_acc2, test_acc3


def optimize_model(model, x, y, x_test, y_test, batch_size=32, learning_rate=1e-4, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    N = y.size(0)
    num_one_epoch = np.floor(N / batch_size).astype(np.int)
    num_epoch = np.floor(3000/num_one_epoch).astype(np.int)
    for epoch in range(num_epoch):
        index = torch.randperm(N)
        for t in range(num_one_epoch):
            idx_start = t*batch_size
            idx_end = (t+1)*batch_size
            y_pred = model(x[index[idx_start:idx_end], :])
            loss = torch.nn.BCELoss()(y_pred, y[index[idx_start:idx_end]])
            # print(epoch, t, loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_pred = model(x)
    loss = torch.nn.BCELoss()(y_pred, y)
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y).float()
    acc = correct.sum() / y.size(0)

    y_test_pred = model(x_test)
    test_loss = torch.nn.BCELoss()(y_test_pred, y_test)
    y_test_pred = (y_test_pred > 0.5).float()
    correct = (y_test_pred == y_test).float()
    test_acc = correct.sum() / y_test.size(0)

    # print(test_loss.data[0])
    print(loss.data[0], acc.data[0], test_loss.data[0], test_acc.data[0])
    return loss.data[0], test_loss.data[0], acc.data[0], test_acc.data[0]


def plot_loss_curve(x, loss1, loss2, loss3, test_loss1, test_loss2, test_loss3, xlabel, ylabel):
    plt.figure()
    plt.plot(x, loss1, 'b-', linewidth=2, label='Train classification loss [x1]')
    plt.plot(x, loss2, 'g-', linewidth=2, label='Train classification loss [x1, x2]')
    plt.plot(x, loss3, 'r-', linewidth=2, label='Train classification loss [x1, x2, noise]')
    plt.plot(x, test_loss1, 'b--', linewidth=2, label='Test classification loss [x1]')
    plt.plot(x, test_loss2, 'g--', linewidth=2, label='Test classification loss [x1, x2]')
    plt.plot(x, test_loss3, 'r--', linewidth=2, label='Test classification loss [x1, x2, noise]')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_acc_curve(x, acc1, acc2, acc3, test_acc1, test_acc2, test_acc3, xlabel, ylabel):
    plt.figure()
    plt.plot(x, acc1, 'b-', linewidth=2, label='Train classification accuracy [x1]')
    plt.plot(x, acc2, 'g-', linewidth=2, label='Train classification accuracy [x1, x2]')
    plt.plot(x, acc3, 'r-', linewidth=2, label='Train classification accuracy [x1, x2, noise]')
    plt.plot(x, test_acc1, 'b--', linewidth=2, label='Test classification accuracy [x1]')
    plt.plot(x, test_acc2, 'g--', linewidth=2, label='Test classification accuracy [x1, x2]')
    plt.plot(x, test_acc3, 'r--', linewidth=2, label='Test classification accuracy [x1, x2, noise]')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def run_exp1():
    N = [128, 256, 512, 800, 1024, 2048]
    D_in = 5
    D_noise = 100
    H = 100
    repeat = 20
    loss1, loss2, loss3, test_loss1, test_loss2, test_loss3 = np.zeros(len(N)), np.zeros(len(N)), \
                                                              np.zeros(len(N)), np.zeros(len(N)), \
                                                              np.zeros(len(N)), np.zeros(len(N))
    acc1, acc2, acc3, test_acc1, test_acc2, test_acc3 = np.zeros(len(N)), np.zeros(len(N)), \
                                                        np.zeros(len(N)), np.zeros(len(N)), \
                                                        np.zeros(len(N)), np.zeros(len(N))
    for i in range(len(N)):
        for n in range(repeat):
            l1, l2, l3, l1_t, l2_t, l3_t, a1, a2, a3, a1_t, a2_t, a3_t = run(N[i], D_in, D_noise, H)
            loss1[i] += l1
            loss2[i] += l2
            loss3[i] += l3
            test_loss1[i] += l1_t
            test_loss2[i] += l2_t
            test_loss3[i] += l3_t
            acc1[i] += a1
            acc2[i] += a2
            acc3[i] += a3
            test_acc1[i] += a1_t
            test_acc2[i] += a2_t
            test_acc3[i] += a3_t
        loss1[i] = loss1[i] / repeat
        loss2[i] = loss2[i] / repeat
        loss3[i] = loss3[i] / repeat
        test_loss1[i] = test_loss1[i] / repeat
        test_loss2[i] = test_loss2[i] / repeat
        test_loss3[i] = test_loss3[i] / repeat
        acc1[i] = acc1[i] / repeat
        acc2[i] = acc2[i] / repeat
        acc3[i] = acc3[i] / repeat
        test_acc1[i] = test_acc1[i] / repeat
        test_acc2[i] = test_acc2[i] / repeat
        test_acc3[i] = test_acc3[i] / repeat

    plot_loss_curve(N, loss1, loss2, loss3, test_loss1, test_loss2, test_loss3, '# train data', 'loss')
    plot_acc_curve(N, acc1, acc2, acc3, test_acc1, test_acc2, test_acc3, '# train data', 'accuracy')


def run_exp2():
    N = 128
    D_in = 5
    D_noise = [10, 20, 30, 50, 80, 100]
    H = 100
    repeat = 20
    loss1, loss2, loss3, test_loss1, test_loss2, test_loss3 = np.zeros(len(D_noise)), np.zeros(len(D_noise)), \
                                                              np.zeros(len(D_noise)), np.zeros(len(D_noise)), \
                                                              np.zeros(len(D_noise)), np.zeros(len(D_noise))
    acc1, acc2, acc3, test_acc1, test_acc2, test_acc3 = np.zeros(len(D_noise)), np.zeros(len(D_noise)), \
                                                        np.zeros(len(D_noise)), np.zeros(len(D_noise)), \
                                                        np.zeros(len(D_noise)), np.zeros(len(D_noise))
    for i in range(len(D_noise)):
        for n in range(repeat):
            l1, l2, l3, l1_t, l2_t, l3_t, a1, a2, a3, a1_t, a2_t, a3_t = run(N, D_in, D_noise[i], H)
            loss1[i] += l1
            loss2[i] += l2
            loss3[i] += l3
            test_loss1[i] += l1_t
            test_loss2[i] += l2_t
            test_loss3[i] += l3_t
            acc1[i] += a1
            acc2[i] += a2
            acc3[i] += a3
            test_acc1[i] += a1_t
            test_acc2[i] += a2_t
            test_acc3[i] += a3_t
        loss1[i] = loss1[i] / repeat
        loss2[i] = loss2[i] / repeat
        loss3[i] = loss3[i] / repeat
        test_loss1[i] = test_loss1[i] / repeat
        test_loss2[i] = test_loss2[i] / repeat
        test_loss3[i] = test_loss3[i] / repeat
        acc1[i] = acc1[i] / repeat
        acc2[i] = acc2[i] / repeat
        acc3[i] = acc3[i] / repeat
        test_acc1[i] = test_acc1[i] / repeat
        test_acc2[i] = test_acc2[i] / repeat
        test_acc3[i] = test_acc3[i] / repeat

    for i in range(len(D_noise)):
        D_noise[i] = D_noise[i] / D_in / 2
    plot_loss_curve(D_noise, loss1, loss2, loss3, test_loss1, test_loss2, test_loss3, 'noise dimension ratio', 'loss')
    plot_acc_curve(D_noise, acc1, acc2, acc3, test_acc1, test_acc2, test_acc3, 'noise dimension ratio', 'accuracy')


if __name__ == '__main__':
    # Exp1: Loss w.r.t. number of training data
    run_exp1()

    # Exp2: Loss w.r.t. number of D_noise
    run_exp2()
