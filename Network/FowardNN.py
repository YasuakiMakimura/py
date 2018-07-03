import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from Neurons import Neurons, Layer, Optimizer
import Activation_Function as F


class Net:
    def __init__(self):
        self.l1 = Layer(lr=1.0,
                        _1=Neurons(2, 3, F.sigmoid),
                        _2=Neurons(2, 3, F.sigmoid),
                        )
        self.out = Layer(lr=1.0,
                         _1=Neurons(6, 1, F.sigmoid))
        self.opt = Optimizer(self.out, self.l1)

    def fw(self, net_inp):
        self.net_inp = net_inp
        l1_o = self.l1.layer_out(self.l1('_1', net_inp),
                                 self.l1('_2', net_inp))
        out_o = self.out.layer_out(self.out('_1', l1_o))
        return l1_o, out_o

    def bp(self, teach):
        self.opt(teach, self.net_inp)


if __name__ == "__main__":
    net = Net()
    net_in = np.array([1, 1])
    t = np.array([0.1])
    o_log = []
    for epi in range(1):
        print(f"epi: {epi}")
        hid_o, out_o = net.fw(net_in)
        net.bp(t)
        print(f"out_o: {out_o}")
        o_log.append(out_o)
        print()
    # plt.plot(range(len(o_log)), o_log)
    # plt.show()
