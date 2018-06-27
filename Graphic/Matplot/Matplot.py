# coding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools as it
from copy import *
import functools
import collections
import pandas as pd
import pickle
from typing import Tuple, List, Iterable, Union, Optional, Sequence, TypeVar, \
    Generic
from operator import add


class Option:
    def __init__(self, ax_variable):
        self.ax = ax_variable

    def axis(self, axis: Union[tuple, list]):
        self.ax.axis(axis)

    def title(self, title_name: str):
        self.ax.set_title(title_name)

    def xlabel(self, xlabel_name='X', fontsize=18):
        self.ax.set_xlabel(xlabel_name, fontsize=fontsize)

    def ylabel(self, ylabel_name='Y', fontsize=18):
        self.ax.set_ylabel(ylabel_name, fontsize=fontsize)

class Func:
    def __init__(self):
        self.setup = Set

    def plot(self):


    def update(self, lines_variable, xdata, ydata):
        for i, y in enumerate(ydata):
            lines_variable[i].set_xdata(xdata[i])
            lines_variable[i].set_ydata(y)



    # def line_label(self, line_name: Union[str, tuple]):
    #     self.ax.set_label(line_name)
    #
    # def xticker_locator(self, minorticker_interval, majorticker_interval):
    #     self.ax.xaxis.set_minor_locator(
    #         ticker.MultipleLocator(minorticker_interval))
    #     self.ax.xaxis.set_major_locator(
    #         ticker.MultipleLocator(majorticker_interval))
    #
    # def yticker_locator(self, minorticker_interval, majorticker_interval):
    #     self.ax.yaxis.set_minor_locator(
    #         ticker.MultipleLocator(minorticker_interval))
    #     self.ax.yaxis.set_major_locator(
    #         ticker.MultipleLocator(majorticker_interval))
    #
    # def grid(self, color='black', linestyle='-'):
    #     self.ax.grid(c=color, linestyle=linestyle)
    #
    # def hline(self, y: Union[int, float], xmin: Union[int, float],
    #           xmax: Union[int, float], color='red', linestyle='-', label=''):
    #     self.ax.hlines(y=y, xmin=xmin, xmax=xmax, colors=color,
    #                    linestyles=linestyle, label=label)
    #
    # def vline(self, x: Union[int, float], ymin: Union[int, float],
    #           ymax: Union[int, float], color='red', linestyle='-', label=''):
    #     self.ax.hlines(x=x, ymin=ymin, ymax=ymax, colors=color,
    #                    linestyles=linestyle, label=label)


class Set:
    def __init__(self, num_fig: Union[str, int], figsize=(6, 4)):
        self.fig = plt.figure(num=num_fig, figsize=figsize)

    def lines(self, ax, len_data, color):
        lines = [None] * len_data
        for i in range(len_data):
            lines[i], = ax.plot([None], [None], c=color[i])
            yield lines[i]

    def x(self, xdata, ydata):
        if xdata is None:
            for y in ydata:
                # print(f'inner y:{y}')
                yield range(len(y))
                # print('x1')
        else:
            for i, y in enumerate(ydata):
                if xdata[i] is None:
                    yield range(len(y))
                else:
                    yield xdata[i]

    def c(self, len_data, color):
        if len(color) == len_data:
            for i in range(len_data):
                yield color[i]

        elif isinstance(color, str):
            for i in range(len_data):
                yield color
        else:
            raise NotImplementedError(f'There is a bug in set_c method')

    def axlim(self, len_data, axlim):#todo 軸の範囲の設定
        if len(axlim) == len_data:
            for i in range(len_data):
                yield axlim[i]
        elif len(axlim) == 4: # todo ここから軸の範囲を決めるところからやる
            try:
                flatten_axlim = [c for inner_c in axlim for c in inner_c]
                # todo 軸の範囲のデータのリストの成分をフラットにする処理
            except:# todo ydataが2つ以上で、axlimの要素が4つのときの場合、つまりすべての軸の範囲を統一したいとき
                for i in range(len_data):
                    yield axlim
            else:
                # todo ydataが例えば3つで、axlimの要素が4つの時の場合、つまり全てのydataに対して軸の範囲のデータを与えるとき
                raise NotImplementedError(f'Since axlim is multiple list, '
                                          f'must be n(axlim):{len(axlim)} == '
                                          f'len_data:{len_data}')


class Plot:
    def __init__(self, num_fig: Union[str, int], axis, axlabel=('', ''),
                 c=None, figsize=(6, 4)):
        self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.setup = Set(num_fig, figsize)
        self.option = Option(self.ax)
        self.option.axis(axis)
        self.option.xlabel(axlabel[0])
        self.option.ylabel(axlabel[1])
        self.color = c
        self.__flag = 0

    def plot(self, xdata=None, ydata=None):
        if self.__flag == 0:
            if xdata is not None:
                if len(xdata) != len(ydata):
                    raise AttributeError(f'For plot(), must be len(xdata):'
                                         f'{len(xdata)} == '
                                         f'len(ydata): {len(ydata)}')
            color = self.setup.c(len(ydata), self.color)
            lines = self.setup.lines(self.ax, len(ydata), list(color))
            self.lines = list(lines)
            self.update(xdata, ydata)
            self.__flag = 1
        else:
            self.update(xdata, ydata)

    def update(self, xdata=None, ydata=None):
        x = self.setup.x(xdata, ydata)
        x = list(x)
        for i, y in enumerate(ydata):
            # print('#############LOOP START################')
            # print('x2')
            # print(f'y:{y}, x:{x}')
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_ydata(y)


class Subplot(Option, Set):
    def __init__(self, num_fig: Union[str, int], nrows=1, ncols=1,
                 figsize=(6, 4), axis=None, axes=None, c=None):
        super().__init__(num_fig, figsize)
        super(Set, self).__init__()
        # self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax = []
        self.nax = nrows * ncols
        self.color = c
        for i in range(self.nax):
            ax_inf = self.fig.add_subplot(nrows, ncols, i+1)
            self.ax.append(ax_inf)
        self.__set_axis(axes=axes)
        self.__flag = 0

    def plot(self, xdata=None, ydata=None):
        if self.__flag == 0:
            color = self.set_c(len(ydata), self.color)
            for i in range(len(ydata)):
                lines = self.set_lines(self.ax[i], len(ydata), list(color))
                self.lines = list(lines)
                self.update(xdata, ydata)
                self.__flag = 0
        else:
            self.update(xdata, ydata)

    def update(self, xdata=None, ydata=None):
        x = self.set_x(xdata, ydata)
        x = list(x)
        for i, y in enumerate(ydata):
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_xdata(y)

    # def __set_lines_list(self, ydata):
    #     self.lines = [None] * len(ydata)
    #     for i, y in enumerate(ydata):
    #         self.lines[i], = self.ax.plot([None], [None],
    #                                              c=self.color[i])
    #
    # def __set_axis(self, axes):
    #     t_axes = np.array(axes)
    #     if t_axes.ndim == 1:
    #         if t_axes.shape == (1, 4):
    #             self.ax[0].axis(axes)
    #     else:
    #         for i, vaxis in range(axes):
    #             self.ax[i].axis(vaxis)

    # def subplt(self, *, xlists=None, ylists):
    #     for i, lines in enumerate(self.lines):
    #         lines[i].set_xdata(range(len(lines)))
    #         lines[i].set_ydata(range(len(lines)))



if __name__ == "__main__":
    p = Plot(num_fig='test', axis=(0, 20, 0, 20), axlabel=('x', 'y'),
             c='black')
    subp = Subplot(num_fig='test', axis=(0, 20, 0, 20),
             c='black')

    # p = Plot(n_line=2, num_fig='test', axis=(0, 50, 0, 50),
    #          axlabel=('x', 'y'), c=('black', 'red'), xrange=False)
    # p.plot(xdata=(None, (3, 4, 5, 6)), ydata=((0, 1, 2, 3), (3, 4, 5, 6)))
    # plt.pause(0.5)
    # p.plot(xdata=((7, 8, 9, 10), (10, 11, 12, 13)),
    #        ydata=((0, 1, 2, 3), (3, 4, 5, 6)))

    # p.plot(ydata=([0, 1, 2, 3],))
    p.plot(xdata=([0, 2, 4, 8], None), ydata=([0, 1, 2, 3], [3, 4, 5, 6, 7, 8]))
    plt.pause(0.5)
    p.plot(ydata=((6, 7, 8, 9), (10, 11, 12, 13)))
    plt.pause(0.5)
    # p.plot(ydata=((0, 1, 2, 3), (3, 4, 5, 6)))
    # plt.show()
    # plt.close()
    # os.exit(0)
