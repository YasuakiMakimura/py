# coding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools as it
from copy import *
import functools
from collections import namedtuple, defaultdict
import pandas as pd
import pickle
from typing import Tuple, List, Iterable, Union, Optional, Sequence, TypeVar, \
    Generic


class Option:
    def __init__(self):
        pass
        # self.ax = ax

    def axis(self, axis: Union[tuple, list]):
        self.ax.axis(axis)

    def title(self, title_name: str):
        self.ax.set_title(title_name)

    def xlabel(self, xlabel_name='X', fontsize=18):
        self.ax.set_xlabel(xlabel_name, fontsize=fontsize)

    def ylabel(self, ylabel_name='Y', fontsize=18):
        self.ax.set_ylabel(ylabel_name, fontsize=fontsize)

    def line_label(self, line_name: Union[str, tuple]):
        self.ax.set_label(line_name)

    def xticker_locator(self, minorticker_interval, majorticker_interval):
        self.ax.xaxis.set_minor_locator(
            ticker.MultipleLocator(minorticker_interval))
        self.ax.xaxis.set_major_locator(
            ticker.MultipleLocator(majorticker_interval))

    def yticker_locator(self, minorticker_interval, majorticker_interval):
        self.ax.yaxis.set_minor_locator(
            ticker.MultipleLocator(minorticker_interval))
        self.ax.yaxis.set_major_locator(
            ticker.MultipleLocator(majorticker_interval))

    def grid(self, color='black', linestyle='-'):
        self.ax.grid(c=color, linestyle=linestyle)

    def hline(self, y: Union[int, float], xmin: Union[int, float],
              xmax: Union[int, float], color='red', linestyle='-', label=''):
        self.ax.hlines(y=y, xmin=xmin, xmax=xmax, colors=color,
                       linestyles=linestyle, label=label)

    def vline(self, x: Union[int, float], ymin: Union[int, float],
              ymax: Union[int, float], color='red', linestyle='-', label=''):
        self.ax.hlines(x=x, ymin=ymin, ymax=ymax, colors=color,
                       linestyles=linestyle, label=label)

class Set:
    def __init__(self, num_fig: Union[str, int], figsize=(6, 4)):
        self.fig = plt.figure(num=num_fig, figsize=figsize)
    
    def set_lines(self, ax, len_data, color):
        lines = [None] * len_data
        for i in range(len_data):
            lines[i], = ax.plot([None], [None], c=color[i])
        # return lines
            yield lines[i]

    def set_lines2(self, xylist):#todo This have not been completed yet
        lines = [None] * len(xylist)
        for i in range(len(ylist)):
            lines[i], = ax.plot([None], [None], c=color[i])
            yield lines[i]

    def set_x(self, xlist, ylist):
        if xlist is None:
            for y in ylist:
                # print(f'inner y:{y}')
                yield range(len(y))
                # print('x1')
        else:
            for i, y in enumerate(ylist):
                if xlist[i] is None:
                    yield range(len(y))
                else:
                    yield xlist[i]

    def set_x2(self, xdata_flag, xydata):#todo 4 This should do for now
        if xdata_flag is None:
            for y in xydata:
                yield range(len(y))
        else:
            for i, xy in enumerate(xydata):
                if xy[0] is None:
                    yield range(len(xy[1]))
                else:
                    yield xy[0]


class Plot(Option, Set):
    def __init__(self, *, num_fig: Union[str, int], axis, axlabel=('', ''),
                 c=None, figsize=(6, 4)):
        super().__init__()
        super(Option, self).__init__(num_fig, figsize)
        # self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis(axis)
        self.ax.set_xlabel(axlabel[0])
        self.ax.set_ylabel(axlabel[1])
        self.colors = c
        self.__flag = 0

    # def __set_lines_list(self, ylist):
    #     self.lines_list = [0] * len(ylist)
    #     for i, y in enumerate(ylist):
    #         self.lines_list[i], = self.ax.plot([0], [0], c=self.colors_list[i])
    # 
    # def __x_range_plot(self, *, xlist=None, ylist):
    #     for i, lines in enumerate(self.lines_list):
    #         lines.set_xdata(range(len(ylist[i])))
    #         lines.set_ydata(ylist[i])
    # 
    # def __plot(self, *, xlist=None, ylist):
    #     if len(xlist) != len(ylist):
    #         raise AttributeError(f'For plot(), must be len(xlist):'
    #                              f'{len(xlist)} == len(ylist): {len(ylist)}')
    #     for i, lines in enumerate(self.lines_list):
    #         if xlist[i] is None:
    #             lines.set_xdata(range(len(ylist[i])))
    #         else:
    #             lines.set_xdata(xlist[i])
    #         lines.set_ydata(ylist[i])
    # 
    # def _plot(self, *, xlist=None, ylist):
    #     self.__lines_list(ylist=ylist)
    #     if xlist is None:
    #         self.plot = self.__x_range_plot
    #     else:
    #         self.plot = self.__plot
    #     return self.plot(xlist=xlist, ylist=ylist)
    
    # def plot(self, xlist=None, ylist=None):
    #     self.lines = self.set_lines(self.ax, ylist, self.colors)
    #     self.__plot(xlist, ylist)
    #     self.plot = self.__plot

    def plot(self, xlist=None, ylist=None):
        if self.__flag == 0:
            if xlist is not None:
                if len(xlist) != len(ylist):
                    raise AttributeError('For plot(), must be len(xlist):'
                                         f'{len(xlist)} == '
                                         f'len(ylist): {len(ylist)}')
            # for i in range(len(ylist)):
            self.lines = self.set_lines(self.ax, len(ylist), self.colors)
            self.lines = list(self.lines)
            self.update(xlist, ylist)
            self.__flag = 1
        else:
            self.update(xlist, ylist)

    def plot2(self, *xy, xdata=None):
        if self.__flag == 0:
            self.lines = self.set_lines(self.ax, len(xy), self.colors)
            self.lines = list(self.lines)
            self.update(xlist, ylist)#todo 1
            self.__flag = 1
        else:
            self.update()#todo This have not been completed yet


    def update(self, xlist=None, ylist=None):
        x = self.set_x(xlist, ylist)
        x = list(x)
        for i, y in enumerate(ylist):
            # print('#############LOOP START################')
            # print('x2')
            # print(f'y:{y}, x:{x}')
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_ydata(y)

    def update2(self, xydata, xdata_flag=None):
        x = self.set_x2(xdata_flag, xydata)#todo 2 This should do for now
        x = list(x)
        for i, xy in enumerate(xydata):
            # print('#############LOOP START################')
            # print('x2')
            # print(f'y:{y}, x:{x}')
            self.lines[i].set_xdata(x[i])
            self.lines[i].set_ydata(xy[1])


class Subplot(Option, Set):# todo This have not been completed yet
    def __init__(self, num_fig: Union[str, int], nrows=1, ncols=1,
                 figsize=(6, 4), axis=None, axes=None):
        super().__init__()
        super(Option).__init__(num_fig=num_fig, figsize=figsize)
        # self.fig = plt.figure(num=num_fig, figsize=figsize)
        self.ax_list = []
        self.nax = nrows * ncols
        for i in range(self.nax):
            sub_inf = self.fig.add_subplot(nrows, ncols, i+1)
            self.ax_list.append(sub_inf)
        self.__set_axis(axes=axes)
        self.__flag = 0

    def plot(self, xlist=None, ylist=None):
        if self.__flag == 0:
            self.lines = self.set_lines(self.ax_list[i], )


    def __set_lines_list(self, ylist):
        self.lines = [None] * len(ylist)
        for i, y in enumerate(ylist):
            self.lines[i] = self.ax_list.plot([None], [None],
                                                 c=self.colors_list[i])

    def __set_axis(self, axes):
        t_axes = np.array(axes)
        if t_axes.ndim == 1:
            if t_axes.shape == (1, 4):
                self.ax_list[0].axis(axes)
        else:
            for i, vaxis in range(axes):
                self.ax_list[i].axis(vaxis)

    def subplt(self, *, xlists=None, ylists):
        for i, lines in enumerate(self.lines):
            lines[i].set_xdata(range(len(lines)))
            lines[i].set_ydata(range(len(lines)))



if __name__ == "__main__":
    p = Plot(num_fig='test', axis=(0, 20, 0, 20), axlabel=('x', 'y'),
             c=('black', 'red'))

    # p = Plot(n_line=2, num_fig='test', axis=(0, 50, 0, 50),
    #          axlabel=('x', 'y'), c=('black', 'red'), xrange=False)
    # p.plot(xlist=(None, (3, 4, 5, 6)), ylist=((0, 1, 2, 3), (3, 4, 5, 6)))
    # plt.pause(0.5)
    # p.plot(xlist=((7, 8, 9, 10), (10, 11, 12, 13)),
    #        ylist=((0, 1, 2, 3), (3, 4, 5, 6)))

    # p.plot(ylist=([0, 1, 2, 3],))
    p.plot(xlist=([0, 2, 4, 8], None), ylist=([0, 1, 2, 3], [3, 4, 5, 6, 7, 8]))
    plt.pause(0.5)
    p.plot(ylist=((6, 7, 8, 9), (10, 11, 12, 13)))
    # p.plot(ylist=((0, 1, 2, 3), (3, 4, 5, 6)))
    plt.show(1)
    # os.exit(0)
