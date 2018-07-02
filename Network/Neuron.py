import numpy as np
import Activation_Function as F


class Neuron:
    def __init__(self, n_in, num, wscale=1.0, bias=0.0, gain=1.0):
        self.n_in = n_in
        self.num = num
        self.bias = bias
        self.gain = gain
        # self.lr = lr
        self.func = None
        self.w = np.random.uniform(-wscale, wscale, (num, n_in))
        # print(f'w: {self.w}')
        # print(f'w.type: {type(self.w)}')
        # print(f'w.shape: {self.w.shape}')
        # self.u = np.zeros(num)
        # self.o = np.zeros(num)


class Layer:
    def __init__(self, lr, **neurons):
        self.neurons = neurons
        self.lr = lr
        self.w = []
        for neuron in neurons.values():
            self.w.append(neuron.w)
        self.w = np.vstack(self.w)
        print(repr(self.w))
        self.o = None
        self.o_diff = []

    def __call__(self, key, inp, func=F.sigmoid):
        try:
            neuron = self.neurons[key]
        except KeyError:
            print(f'"neurons" have no key "{key}"')
        else:
            neuron.func = func
            u = np.inner(neuron.w, inp)
            o = np.fromiter((func(u, neuron.gain, neuron.bias)), np.float)
            self.o_diff.append(
                np.fromiter((func(u, neuron.gain, neuron.bias, diff=True)), np.float)
            )
            return o

    def layer_out(self, *neurons_output):
        def generator_neuron_output():
            for neuron_output in neurons_output:
                yield neuron_output

        self.o = np.ravel(list(generator_neuron_output()))
        self.o_diff = np.ravel(self.o_diff)
        return self.o

    # def update(self, key, net_in, error):
    #     # 誤差求める、誤差信号求める、重み値の更新量
    #     # 誤差... 教師、層の出力
    #     # 誤差信号...出力層の場合 誤差、出力の微分
    #     #           中間層の場合 出力層の誤差信号、重み値、中間層の出力の微分


class Optimizer:
    def __init__(self, out_layer, *hid_layer):
        self.out_layer = out_layer
        self.hid_layer = hid_layer

    def update(self, teach):
        def error():
            return teach - self.out_layer.o

        self.out_layer.err_signal = error() * self.out_layer.o_diff
        self.out_layer.w \
            += self.out_layer.lr * np.matrix(self.out_layer.err_signal).T * self.hid_layer[0].o
        next_layer = self.out_layer
        for now_layer in self.hid_layer:
            print(np.matrix(next_layer.err_signal))
            print()
            print(next_layer.w)
            print()
            print(repr(np.matrix(next_layer.err_signal) * next_layer.w))
            print()
            print(now_layer.o_diff)
            err_signal_w = np.matrix(next_layer.err_signal) * next_layer.w
            # now_layer.err_signal =  * now_layer.o_diff
        # print(f"out-layer err_signal: {self.out_layer.err_signal.shape}")
        # print(f'err_signal(out): {self.out.err_signal}')


if __name__ == "__main__":
    ...
