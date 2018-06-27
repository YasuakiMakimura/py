# import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def lines_set(ax, len_data, color):
    lines = [None] * len_data
    for i in range(len_data):
        lines[i], = ax.plot([None], [None], c=color[i])
        yield lines[i]


def lines_set2(ax, len_data, color):
    lines = [None] * len_data
    for i in range(len_data):
        lines[i], = ax[i].plot([None], [None], c=color[i])
        yield lines[i]


def c_set(ydata, color):
    len_data = len(ydata)
    if color is None:
        for i in range(len_data):
            yield None
    elif len(color) == len_data:
        for i in range(len_data):
            yield color[i]
    elif isinstance(color, str):
        for i in range(len_data):
            yield color
    else:
        data_size_check(ydata, color)


def x_set(xdata, ydata):
    if xdata is None:
        if ydata == flatten(ydata):
            yield range(len(ydata))
        else:
            for i, y in enumerate(ydata):
                yield range(len(y))
    else:
        for i, y in enumerate(ydata):
            if xdata[i] is None:
                yield range(len(y))
            else:
                yield xdata[i]


def data_size_check(xdata=None, ydata=None):
    if len(xdata) != len(ydata):
        print(f'xdata: {xdata}')
        print(f'ydata: {ydata}')
        raise ValueError(f'shape mismatch of xdata and ydata: '
                         f'len xdata: {len(xdata)}, '
                         f'len ydata: {len(ydata)}')


def flatten(L):
    if isinstance(L, list):
        # noinspection PySimplifyBooleanCheck
        if L == []:
            return []
        else:
            return flatten(L[0]) + flatten(L[1:])
    else:
        return [L]


class Option:
    pass


class Plot:
    def __init__(self, num_fig: Union[str, int], axis, axlabel=('', ''), figsize=(6, 4), c=None):
        self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis(axis)
        self.ax.set_xlabel(axlabel[0])
        self.ax.set_ylabel(axlabel[1])
        self.c = c
        self.__flag = 0

    def __set(self, xdata=None, ydata=None):
        if xdata is not None:
            data_size_check(xdata, ydata)
        color = c_set(ydata, self.c)
        color = list(color)
        lines = lines_set(self.ax, len(ydata), color)
        self.lines = list(lines)

    def plot(self, xdata=None, ydata=None):
        if self.__flag == 0:
            self.__set(xdata, ydata)
            self.update(xdata, ydata)
            self.__flag = 1
        else:
            self.update(xdata, ydata)

    def update(self, xdata=None, ydata=None):
        x = list(x_set(xdata, ydata))
        for i, y in enumerate(ydata):
            data_size_check(x[i], y)
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_ydata(y)


class Subplot:
    def __init__(self, num_fig: Union[str, int], axis, figsize=(6, 4), nrows=1, ncols=1, c=None):
        self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax = []
        self.nax = nrows * ncols
        self.c = c
        for i in range(self.nax):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.axis(axis)
            self.ax.append(ax)
        self.__flag = 0

    def __set(self, ydata=None):
        color = c_set(ydata, self.c)
        color = list(color)
        lines = lines_set2(self.ax, len(ydata), color)
        self.lines = list(lines)
        print(self.lines)

    def plot(self, xdata=None, ydata=None):
        if self.__flag == 0:
            self.__set(ydata)
            self.update(xdata, ydata)
            self.__flag = 1
        else:
            self.update(xdata, ydata)

    def update(self, xdata=None, ydata=None):
        x = list(x_set(xdata, ydata))
        for i, y in enumerate(ydata):
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_ydata(y)


if __name__ == "__main__":
    # p = Plot(num_fig='test', axis=(0, 20, 0, 20), axlabel=('x', 'y'), c=('black', 'red'))
    # p.plot(xdata=(None, (3, 4, 5, 6)), ydata=([0, 1, 2, 3], [3, 4, 5, 6]))
    # p.plot(xdata=((3, 4, 5, 6), None), ydata=([0, 1, 2, 3], [3, 4, 5, 6, 7]))

    # p = Plot(num_fig='test', axis=(0, 20, 0, 20), axlabel=('x', 'y'), c=None)
    # p.plot(ydata=([0, 1, 2, 3], [3, 4, 5, 6, 7], [7, 8, 9], [10, 11]))

    sp = Subplot(num_fig='test(subplot)', axis=(0, 20, 0, 20), nrows=1, ncols=3,
                 c=('red', 'black', 'magenta'))
    sp.plot(xdata=None, ydata=([1, 2, 3, 4], [10, 11, 12, 13], [1, 1, 1, 1]))

    plt.pause(2)

# todo もっと分解して作る
# todo yieldが本当にいるのか再検証がいる
# todo method もう少し分解してもいいかも
# todo Plot Subplotクラスの上にそれぞれクラスを作ってそれをPlot Subplotにインスタンスするかどうか再検討
