import numpy as np
import matplotlib.pyplot as plt


def sigmoid(internal_state: np.ndarray, gain=1.0, bias=0.0, differential=False):
    if differential:
        output = 1 / (1 + np.exp(-(internal_state * gain) + bias))
        return output * (1 - output)
    return 1 / (1 + np.exp(-(internal_state * gain) + bias))


def tanh(internal_state: np.ndarray, gain=1.0, bias=0.0, differential=False):
    if differential:
        output = np.tanh(gain * internal_state + bias)
        return 1.0 / (np.cosh(output) ** 2)
    return np.tanh(gain * internal_state + bias)
