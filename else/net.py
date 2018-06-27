# ^*^ coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
import func as fn


class Neuron:
    def __init__(self, num, n_in, tau=1.0, bias=0.0, rec=False, func=fn.Tanh, gain=1.0):
        self.u = np.zeros(num)
        self.o = np.zeros(num)
        self.w = np.random.uniform(-1, 1, (num, n_in))
        self.w_bu = self.w.copy()  # to backup weights
        self.bw = np.random.uniform(-1, 1, n_in)
        self.delta = np.zeros(num)
        self.u_old = np.zeros(num)
        self.o_old = np.zeros(num)
        self.del_old = np.zeros(num)
        self.rec = rec
        self.num = num
        self.n_in = n_in
        self.tau = tau
        self.bias = bias
        self.fn = func
        self.gain = gain
        if rec:
            self.rw = np.random.uniform(-1, 1, (num, num))
            self.rw_bu = self.rw.copy()  # to backup rec. weights

    def uniform_w(self, w_min, w_max):
        self.w = np.random.uniform(w_min, w_max, (self.num, self.n_in))

    def uniform_rw(self, w_min, w_max):
        self.rw = np.random.uniform(w_min, w_max, (self.num, self.num))

    def normal_w(self, w_mean, w_var):
        self.w = np.random.normal(w_mean, w_var, (self.num, self.n_in))

    def normal_rw(self, w_mean, w_var):
        self.rw = np.random.normal(w_mean, w_var, (self.num, self.num))

    def forward(self, l_in):
        self.u_old = self.u.copy()
        self.o_old = self.o.copy()
        self.u = np.zeros(self.num)
        self.l_in = l_in
        self.u = self.w.dot(l_in)
        if self.rec:
            self.u += self.rw.dot(self.o_old)
        if self.bias:
            self.u += self.bias
        if self.tau > 1.0:
            lam = (1.0 / self.tau)
            self.u *= lam
            self.u += (1.0 - lam) * self.u_old
        self.o = self.fn(self.u, gain=self.gain)

    def feedback(self, delta, weight):
        self.delta = delta.dot(weight)
        self.delta *= self.fn(self.u, gain=self.gain, diff=True)

    def update_w(self, eta=1.0):
        dw = self.delta.reshape(self.num, 1) * self.l_in
        dw *= eta
        self.w += dw

    def update_rw(self, eta=1.0):
        dw = self.delta.reshape(self.num, 1) * self.o_old
        dw *= eta
        self.rw += dw


# class Net:
#     def __init__(self, **neurons):
#         self.layers = len(neurons)
#         for name, neuron in neurons.items():
#             self.add_neuron(name, neuron)
#         self.set_arc(neurons)
#
#     def add_neuron(self, name, neuron):
#         if name in self.__dict__:
#             raise AttributeError('cannot register a new neuron %s: attribute exists' % name)
#         else:
#             self.__dict__[name] = neuron
#
#     def set_arc(self, *neurons):
#         self.arc = neurons


class Net:
    def __init__(self, keys, *neurons):
        if len(keys) != len(neurons):
            print('ERROR!', 'NOT len(keys): {} = len(neurons): {}'.format(len(keys), len(neurons)))
            sys.exit(1)
        self.layers = len(neurons)
        for neuron in range(len(neurons)):
            self.add_neuron(keys[neuron], neurons[neuron])
        self.neurons = neurons

    def __call__(self, v_in):
        self.v_in = v_in
        self.neurons[0].forward(v_in)
        for layer in range(self.layers - 1):
            self.neurons[layer + 1].forward(self.neurons[layer].o)
        return self.neurons[self.layers - 1].o

    def add_neuron(self, name, neuron):
        if name in self.__dict__:
            raise AttributeError('cannot register a new neuron {}: attribute exists'.format(name))
        else:
            self.__dict__[name] = neuron

    def update(self, target):
        neuron = self.neurons
        readout = self.neurons[self.layers - 1]
        readout.delta = target - readout.o
        df = readout.fn(readout.u, gain=readout.gain, diff=True)
        readout.delta *= df
        for layer in range(self.layers - 1, 0, -1):
            neuron[layer].update_w()
            neuron[layer - 1].feedback(neuron[layer].delta, neuron[layer].w)
        neuron[0].update_w(self.v_in)


if __name__ == '__main__':
    np.random.seed(1)
    EPISODE = 1000
    plt.xlim(0, EPISODE)
    plt.xlabel("epoch")
    plt.ylim(-1, 1)
    plt.yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    cc = ["blue", "red", "magenta", "cyan"]
    le = ["[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"]
    po = [[] for j in range(4)]
    # ex_in = [[0, 0, 0.1], [0, 1, 0.1], [1, 0, 0.1], [1, 1, 0.1]]
    # ex_t = [-0.8, 0.8, 0.8, -0.8]
    # h = Neuron(3, 3)
    # h.uniform_w(-1, 1)
    # o = Neuron(1, 3)
    # o.uniform_w(-1, 1)
    # for ep in range(EPISODE):
    #     sys.stdout.write("\n{0}".format(str(ep)))
    #     for pp in range(4):
    #         h.forward(np.tanh, ex_in[pp])
    #         o.forward(np.tanh, h.o)
    #         sys.stdout.write("\t{0}".format(str(o.o)))
    #         po[pp].append(o.o)
    #         o.delta = ex_t[pp] - o.o
    #         o.update_w(h.o)
    #         h.feedback((1.0 - h.o*h.o), o.delta, o.w)
    #         h.update_w(ex_in[pp])
    # for pp in range(4):
    #     plt.plot(range(EPISODE), po[pp], label=le[pp], color=cc[pp])
    # plt.legend()
    # plt.savefig('fig_net')
    # plt.show()
    #
    # test = Net(x=Neuron(3, 2), o=Neuron(1, 3), h=Neuron(5, 2))
    # # test.set_arc(test.x, test.o)
    # print('class: ', test)
    # test.x.uniform_w(-1, 1)
    # print('layers: ', test.layers)
    # print('h: ', test.x)
    # print('o: ', test.o)
    # print('\t\t\t\t\t ↕︎')
    # print('arc: ', test.arc)
    # print(test.x.w)
    # print(test.arc)

    net = Net(('h', 'o'), Neuron(5, 3), Neuron(1, 5))
    x = np.array([[0, 0, 0.1], [0, 1, 0.1], [1, 0, 0.1], [1, 1, 0.1]])
    t = np.array([-0.8, 0.8, 0.8, -0.8])
    for ep in range(EPISODE):
        sys.stdout.write("\n{0}".format(str(ep)))
        for pp in range(4):
            net(x[pp])
            net.update(t[pp])
            sys.stdout.write("\t{0}".format(str(net.o.o)))
            po[pp].append(net.o.o)
    for pp in range(4):
        plt.plot(range(EPISODE), po[pp], label=le[pp], color=cc[pp])
    plt.legend()
    plt.show()
