import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Union


def wrap(gain):
    if gain <= 0:
        raise ValueError('Not gain <= 0')
    return gain


def sigmoid(u: Union[list, tuple, np.ndarray], gain=1.0, bias=0.0, diff=False):
    gain = wrap(gain)
    for i, u_value in enumerate(u):
        o = 1 / (1 + math.exp(-gain * u_value))
        if diff:
            yield gain * o * (1 - o)
        yield o


def tanh(u: Union[list, tuple, np.ndarray], gain=1.0, bias=0.0, diff=False):
    gain = wrap(gain)
    for i, u_value in enumerate(u):
        if diff:
            yield gain / (math.cosh(gain * u_value) ** 2)
        o = math.tanh(gain * u_value)
        yield o


def relu(u: Union[list, tuple, np.ndarray], gain=1.0, bias=0.0, diff=False):
    gain = wrap(gain)
    if diff:
        for i, u_value in enumerate(u):
            if u_value <= -bias:
                yield 0.0
            else:
                yield gain
    else:
        for i, u_value in enumerate(u):
            if u_value <= -bias:
                yield 0.0
            else:
                yield gain * (u_value + bias)


def reth(u: Union[list, tuple, np.ndarray], gain=1.0, bias=0.0, diff=False):
    gain = wrap(gain)
    if diff:
        for i, u_value in enumerate(u):
            if u_value <= -bias:
                yield 0.0
            else:
                yield gain / (math.cosh(gain * (u_value + bias)) ** 2)
    else:
        for i, u_value in enumerate(u):
            if u_value <= -bias:
                yield 0.0
            else:
                o = math.tanh(gain * u_value + bias)
                yield o


# class Func:
#     # __slots__ = ['sigmoid', 'tanh', 'relu', 'reth']
#
#     @property
#     def sigmoid(self):
#         return sigmoid
#
#     @property
#     def tanh(self):
#         return tanh
#
#     @property
#     def relu(self):
#         return relu
#
#     @property
#     def reth(self):
#         return reth


if __name__ == '__main__':
    u = np.arange(-10, 10, 0.1)
    # # plt.grid()
    # # plt.plot(u, np.array(list(sigmoid(u))))
    # # plt.show()
    # # plt.grid()
    # # plt.plot(u, np.array(list(tanh(u))))
    # # plt.show()
    # # plt.grid()
    # # plt.plot(u, list(relu(u)))
    # # plt.show()
    # plt.grid()
    # plt.ylim(0.9, 1.0)
    # plt.xlim(2.2, 2.5)
    # plt.plot(u, list(reth(u)))
    # plt.show()
