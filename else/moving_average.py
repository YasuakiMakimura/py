import numpy as np
import matplotlib.pyplot as plt


def ma(data, len_run):
    run = np.ones(len_run)/len_run
    return np.convolve(data, run, mode='same')


if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
    yorg = np.sin(x)
    y = np.sin(x) + np.random.randn(100) * 0.2

    d = ma(y, 5)

    plt.plot(x, yorg, 'r', label='オリジナルsin')
    plt.plot(x, y, 'k-', label='元系列')
    plt.plot(x, d, 'b--', label='移動平均')
    plt.legend()
    plt.show()



