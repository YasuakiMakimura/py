import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from Neuron import Neuron, Layer, Optimizer
import Activation_Function as F


class Net:
    def __init__(self):
        self.l1 = Layer(_1=Neuron(2, 3),
                        _2=Neuron(2, 3))
        self.out = Layer(_1=Neuron(6, 1))
        self.opt = Optimizer(self.out, self.l1)

    def fw(self, net_inp):
        l1_o = self.l1('_1', net_inp, func=F.sigmoid) + \
               self.l1('_2', net_inp, func=F.tanh)
        # self.l1.o = np.ravel([self.l1('_1', net_inp, func=F.sigmoid),
        #                       self.l1('_2', net_inp, func=F.tanh)])
        self.out.o = self.out('_1', self.l1.o, func=F.sigmoid)
        return self.l1.o, self.out.o

    def bp(self, teach):
        self.opt.update(teach)


if __name__ == "__main__":
    net = Net()
    net_in = np.array([1, 1])
    t = np.array([0.1])
    hid_o, out_o = net.fw(net_in)
    # print(top_o)
    net.bp(t)
