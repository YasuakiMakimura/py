import numpy as np
import Activation_Function as F


class Neuron:
    def __init__(self, n_in, num, wscale=1.0, bias=0.0, gain=1.0, lr=1.0):
        self.n_in = n_in
        self.num = num
        self.bias = bias
        self.gain = gain
        self.lr = lr
        self.func = None
        self.w = np.random.uniform(-wscale, wscale, (num, n_in))
        # print(f'w: {self.w}')
        # print(f'w.type: {type(self.w)}')
        # print(f'w.shape: {self.w.shape}')
        # self.u = np.zeros(num)
        # self.o = np.zeros(num)


class Layer:
    def __init__(self, **neurons):
        self.neurons = neurons
        self.o = None

    def __call__(self, key, inp, func=F.sigmoid):
        try:
            neuron = self.neurons[key]
        except KeyError:
            print(f'"neurons"  have no key "{key}"')
        else:
            neuron.func = func
            neuron.u = np.inner(neuron.w, inp)
            neuron.o = np.fromiter((func(neuron.u, neuron.gain)), np.float)
            self.diff \
                = np.fromiter((func(neuron.u, neuron.gain, diff=True)), np.float)
            return neuron.o

    def __add__(self, other):
        return np.ravel()

    # def update(self, key, net_in, error):
    #     # 誤差求める、誤差信号求める、重み値の更新量
    #     # 誤差...　教師、層の出力
    #     # 誤差信号... 出力層の場合　誤差、出力の微分
    #     #             中間層の場合　出力層の誤差信号、重み値、中間層の出力の微分
    #     try:
    #         neuron = self.neurons[key]
    #     except KeyError:
    #         print(f'"neurons" have no key "{key}"')
    #     else:
    #         err_sig \
    #             =  np.array(list(self.func(neuron.u, neuron.gain, diff=True)))
    #         neuron.w += neuron.lr * err_sig * net_in


class Optimizer:
    def __init__(self, out_layer, *hid_layer):
        self.out = out_layer

    def update(self, teach):
        print(f'teach: {teach}')
        print(f'out layer output: {self.out.o}')
        def error():
            return teach - self.out.o
        print(f'error: {error()}')
        self.out.err_signal = error() * self.out.diff
        print(f'err_signal(out): {self.out.err_signal}')


if __name__ == "__main__":
    ...
