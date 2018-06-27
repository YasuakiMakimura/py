#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np


# noinspection PyMissingOrEmptyDocstring
def sigmoid(val_u, gain=1.0, diff=False):
    if diff:
        o = 1.0 / (1.0 + np.exp(-val_u))
        return o * (1.0 - o)

    return 1.0 / (1.0 + np.exp(-gain * val_u))


# noinspection PyMissingOrEmptyDocstring
def tanh(val_u, gain=1.0, diff=False):
    if diff:
        o = np.tanh(gain * val_u)
        return 1.0 / (np.cosh(o)**2)

    return np.tanh(gain * val_u)


# noinspection PyMissingOrEmptyDocstring
def reth(val_u, gain=1.0, diff=False):
    if diff:
        if val_u < 0.0:
            return 0.0
        else:
            o = np.tanh(gain * val_u)
            return 1.0 / (np.cosh(o)**2)
    else:
        if val_u < 0.0:
            return 0.0
        else:
            return np.tanh(gain * val_u)


# noinspection PyMissingOrEmptyDocstring
def relu(val_u, gain=1.0, diff=False):
    if diff:
        if val_u < 0.0:
            return 0.0
        else:
            return 1.0
    else:
        if val_u < 0.0:
            return 0.0
        else:
            return gain * val_u


if __name__ == "__main__":
    pass
