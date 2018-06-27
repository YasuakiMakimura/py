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
from typing import Tuple, List, Iterable, Union, Optional, Sequence


def axis(ax, kw):
    return ax.axis(kw['axis'])

def title(ax, kw):
    return ax.set_title(kw['title'])

def xlabel(ax, kw):
    return ax.set_xlabel(kw['xlabel'], fontsize=18)

def ylabel(ax, kw):
    return ax.set_ylabel(kw['ylabel'], fontsize=18)

def line_label(ax, kw, iii):
    return ax.set_label(kw['legend'][iii])

def xticker_locator(ax, kw):
    return {ax.xaxis.set_minor_locator(ticker.MultipleLocator(kw['xticks_minor'])),
            ax.xaxis.set_major_locator(ticker.MultipleLocator(kw['xticks_major']))}

def yticker_locator(ax, kw):
    return {ax.yaxis.set_minor_locator(ticker.MultipleLocator(kw['yticks_minor'])),
            ax.yaxis.set_major_locator(ticker.MultipleLocator(kw['yticks_major']))}

def grid(ax, kw):
    return ax.grid()

def hline(ax, kw):
    return ax.hlines(kw['val_hline'], kw['axis'][0], kw['axis'][1], lw=1)

def vline(ax, kw):
    return ax.vlines(kw['val_vline'], kw['axis'][2], kw['axis'][3], lw=1)

def linestyle(ax, kw, iii):
    return ax.set_linestyle(kw['linestyle'][iii])

def linecolor(n_line, kw):
    if isinstance(kw['c'], (list, tuple)):
        return kw['c']
    elif isinstance(kw['c'], str):
        return [kw['c']] * n_line

def alpha(ax, kw, iii):
    return ax.set_alpha(kw['alpha'][iii])


def setup(ax, kw, target):
    def fig():
        axis(ax, kw)
        title(ax, kw)
        if kw['xlabel'] is not None:
            xlabel(ax, kw)
        if kw['ylabel'] is not None:
            ylabel(ax, kw)
        if [None, None] == [kw['xticks_minor'], kw['xticks_major']]:
            xticker_locator(ax, kw)
        if [None, None] == [kw['yticks_minor'], kw['yticks_major']]:
            yticker_locator(ax, kw)
        if kw['grid']:
            grid(ax, kw)
        if kw['val_hline'] is not None:
            hline(ax, kw)
        if kw['val_vline'] is not None:
            vline(ax, kw)

    def line(var_loop):
        if kw['legend'] is not None:
            line_label(ax, kw, iii=var_loop)
        if kw['alpha'] is not None:
            alpha(ax, kw, iii=var_loop)
        if kw['linestyle'] is not None:
            linestyle(ax, kw, iii=var_loop)

    if target == 'fig':
        return fig()
    elif target == 'line':
        return line
    else:
        raise SyntaxError(f'there is not "target" in argument of function "{setup}"')

    # ax.axis(kw['axis'])
    # ax.set_title(kw['title'])
    # ax.set_xlabel(kw['xlabel'], fontsize=18)
    # ax.set_ylabel(kw['ylabel'], fontsize=18)
    # if [None, None] == [kw['xticks_minor'], kw['xticks_major']]:
    #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(kw['xticks_minor']))
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(kw['xticks_major']))
    # if [None, None] == [kw['yticks_minor'], kw['yticks_major']]:
    #     ax.yaxis.set_minor_locator(ticker.MultipleLocator(kw['yticks_minor']))
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(kw['yticks_major']))
    # if kw['grid']:
    #     ax.grid()
    # if kw['val_hline'] is not None:
    #     ax.hlines(kw['val_hline'], kw['axis'][0], kw['axis'][1], lw=1)
    # if kw['val_vline'] is not None:
    #     ax.vlines(kw['val_vline'], kw['axis'][2], kw['axis'][3], lw=1)
    # ax.tight_layout()

def setup_subplot(ax, kw, target):
    def fig(figure):
        setup(ax, kw, target='fig')
        if kw['axlabel_L']:
            if [None, None] == [kw['ylabel'], kw['xlabel']]:
                figure.text(0.02, 0.5, 'y_axis', ha='center', va='center', rotation='vertical')#todo ここ未完成　x軸のラベルができていない
            else:
                raise TypeError(f'Because kwarg "axlabel_L" is True, both of label of xaxis and yaxis must be None ')

    def line():
        setup(ax, kw, target='line')

    if target == 'fig':
        return fig
    elif target == 'line':
        return line
    else:
        raise SyntaxError(f'there is not "target=={target}" in argument of function "{setup_subplot}"')





def setup_multi(ax, **kw):
    for iii in range(len(ax)):
        ax[iii].axis(kw['axis'][iii])
        ax[iii].set_title(kw['title'][iii])
        ax[iii].set_xlabel(kw['xlabel'][iii], fontsize=18)
        ax[iii].set_ylabel(kw['ylabel'][iii], fontsize=18)
        if [None, None] == [kw['xticks_minor'], kw['xticks_major']]:
            ax[iii].xaxis.set_minor_locator(ticker.MultipleLocator(kw['xticks_minor']))
            ax[iii].xaxis.set_major_locator(ticker.MultipleLocator(kw['xticks_major']))
        if [None, None] == [kw['yticks_minor'], kw['yticks_major']]:
            ax[iii].yaxis.set_minor_locator(ticker.MultipleLocator(kw['yticks_minor']))
            ax[iii].yaxis.set_major_locator(ticker.MultipleLocator(kw['yticks_major']))
        if kw['grid'][iii] or kw['grid']:
            ax[iii].grid()
        if kw['val_hline'] is not None:
            ax[iii].hlines(kw['val_hline'][iii], kw['axis'][0], kw['axis'][1], lw=1)
        if kw['val_vline'] is not None:
            ax[iii].vlines(kw['val_vline'][iii], kw['axis'][2], kw['axis'][3], lw=1)

    if isinstance(kw['xlabel'], (list, tuple)):
        for iii in range(len(ax)):
            ax[iii].set_xlabel(kw['xlabel'][iii], fontsize=18)
            ax[iii].set_ylabel(kw['ylabel'][iii], fontsize=18)
    else:
        ax[-1].set_xlabel(kw['xlabel'], fontsize=18)
        ax[int(len(ax) / 2)].set_ylabel(kw['ylabel'], fontsize=18)



# noinspection PyTypeChecker
class Line:
    def __new__(cls, *args, **kwargs):
        cls.def_kwg = dict(figsize=(6, 4),
                           subplt_arrange=None,
                           c=None,
                           title='',
                           axis=(0, 50, 0, 1.0),
                           x_low=0, x_high=50,
                           y_low=0, y_high=1.0,
                           xlabel='x', ylabel='y',
                           xticks_minor=None, xticks_major=None,
                           yticks_minor=None, yticks_major=None,
                           grid=False, legend=None, alpha=None,
                           val_hline=None, val_vline=None,
                           linestyle=None,
                           axlabel_L=False,
                           inf=None,
                           auto_xrange=True,
                           scatter=False)

        for key, val in kwargs.items():
            try:
                cls.def_kwg[key]
            except KeyError:
                print(f'List of kwargs: {cls.def_kwg.keys()}\n\n')
                raise KeyError(f'"{key}" does not exist in keyword argument of {cls.__class__.__name__}')
            else:
                cls.def_kwg[key] = val
        return super().__new__(cls)

    def __init__(self, num_fig, n_line=1, axis=None, ):
        self.kw = self.def_kwg
        self.fig = plt.figure(num=num_fig, figsize=self.kw['figsize'])
        self.n_line, self.line = n_line, [0] * n_line
        if self.kw['subplt_arrange'] is not None:#todo 複数のaxの場合、ここを明日する
            ax = []
            for iii in range(n_line):
                ax.append(self.fig.add_subplot(self.kw['subplt_arrange'][0], self.kw['subplt_arrange'][1], iii + 1))
                setup_subplot(ax=ax[iii], kw=self.kw, target='fig')(figure=self.fig)
                self.line[iii], = ax[iii].plot([0], [0], c=linecolor(n_line, self.kw))
                setup_subplot(ax=ax[iii], kw=self.kw, target='line')



        else:
            ax = self.fig.add_subplot(111)
            setup(ax=ax, kw=self.kw, target='fig')
            for iii in range(n_line):
                self.line[iii], = ax.plot([0], [0], c=linecolor(n_line, self.kw))
                setup(ax=self.line[iii], kw=self.kw, target='line')(var_loop=iii)
            plt.legend(fontsize=15)
        plt.tight_layout()#todo ここは単一のaxの場合、おそらくできたデバックはしてない
        plt.pause(0.01)

    def single_plot(self, log: Sequence):
        self.line[0].set_data(range(len(log)), log)

    def multi_plot(self, logs: Tuple[Sequence, ...]):
        for iii in range(len(logs)):
            self.lines[iii].set_data(range(len(logs[iii])), logs[iii])





if __name__ == "__main__":
    pass
