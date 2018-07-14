# coding=utf-8

import numpy as np
improt random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, *hid, n_out):
        super().__init__()
        self.fc_in = nn.Linear(n_in, hid[0])
        w = np.random.uniform(-0.1, 0.1, (hid[0], n_in))
        print(w)
        w = torch.FloatTensor(w)
        print(w.type())
        self.fc_in.weight.data = w
        print(self.fc_in.state_dict())

        self.fc_hid = []
        for i, num in enumerate(hid[1:]):
            self.fc_hid.append(nn.Linear(hid[i - 1], num))

        self.fc_out = nn.Linear(hid[-1], n_out)

    def forward(self, x):
        x = F.tanh(self.fc_in(x))
        # yield x
        for i, hid in enumerate(self.fc_hid):
            x = F.tanh(hid(x))
            # yield x
        x = F.tanh(self.fc_out(x))
        # yield F.tanh(self.fc_out(x))
        return x


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    result, = ax.plot([0], [0])

    loss_log = []
    out = [[] for i in range(4)]

    model = Net(2, 3, n_out=1)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    inputs = torch.Tensor([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

    for epoch in range(1):
        print(f'ep: {epoch}')
        running_loss = 0.0

        for pp, inp in enumerate(inputs):
            if [inp[0].item(), inp[1].item()] == [0, 0] or \
                    [inp[0].item(), inp[1].item()] == [1, 1]:
                teach = torch.Tensor([-0.9])
            else:
                teach = torch.Tensor([0.9])

            optimizer.zero_grad()
            outputs = model(inp)

            print(outputs)

            loss = loss_fn(outputs, teach)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            out[pp].append(outputs.item())

        print(f'loss: {running_loss}')
        loss_log.append(running_loss)

    plt.grid()
    plt.yticks([-0.9, 0, 0.9])
    plt.plot(out[0])
    plt.plot(out[1])
    plt.plot(out[2])
    plt.plot(out[3])
    plt.show()
    plt.plot(loss_log)
    plt.show()
    print('Finish Learning')


if __name__ == "__main__":
    main()
