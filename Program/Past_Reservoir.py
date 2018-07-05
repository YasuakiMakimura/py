#!/usr/bin/python3
# -*- coding:utf-8 -*-

import functools
import math
import sys
from copy import *
from typing import Tuple, List, Union, ClassVar
# from numba import jit

import matplotlib.pyplot as plt
import numpy as xp

# import FunctionGenerator_ReNN as fg
# import RealTimePlot_ReNN as rtp
# import func as fn
from time import perf_counter
from pprint import pprint

print(__name__)
# noinspection PyArgumentList,PyMissingOrEmptyDocstring,PyAttributeOutsideInit,PyMethodParameters
class Reservoir:
    def __init__(self, num_in: int, num: int, num_out: int, v_tau, n_tau,
                 p_connect: int = 10, v_lambda: float = 1.8, in_wscale=1.0, fb_wscale=1.0):
        if len(v_tau) != len(n_tau):
            raise AttributeError(f"len(v_tau): {len(v_tau)} != len(n_tau): {len(n_tau)}")
        if sum(n_tau) != num:  # type: Tuple[int], Tuple[int]
            raise AttributeError(f"len: {sum(n_tau)} != len: {num}")

        self.num_in: int = num_in
        self.num: int = num
        self.num_out: int = num_out
        self.p_connect: int = p_connect

        self.v_lambda: float = v_lambda
        self.v_tau: Tuple[float] = v_tau
        self.n_tau: Tuple[float] = n_tau

        inf_con = xp.array(
            xp.random.choice(2, (num, num_in), p=[(100 - p_connect) / 100, p_connect / 100]))
        self.in_w = xp.random.uniform(-in_wscale, in_wscale, (num, num_in))
        self.in_w *= inf_con

        inf_con = xp.array(
            xp.random.choice(2, (num, num), p=[(100 - p_connect) / 100, p_connect / 100]))
        self.rw \
            = v_lambda * xp.random.normal(0,
                                          1 / math.sqrt(float(num * p_connect / 100)), (num, num))
        self.rw *= inf_con
        self.read_w = xp.random.normal(0, 1 / math.sqrt(float(num)), (num_out, num))
        inf_con = xp.array(
            xp.random.choice(2, (num, num_out), p=[(100 - p_connect) / 100, p_connect / 100]))
        self.fb_w = xp.random.uniform(-fb_wscale, fb_wscale, (num, num_out))
        self.fb_w *= inf_con
        self.ru = xp.random.uniform(-1.0, 1.0, num)
        self.old_ru: xp.ndarray
        self.bu_ru: xp.ndarray
        self.in_o = xp.zeros(num_in)
        self.ro = xp.tanh(self.ru)
        self.pre_ro: xp.ndarray
        self.read_o = xp.zeros(num_out)
        # print(self.in_w)
        # print(self.rw)
        # print('init_rw: \n' + f'{self.rw}')
        # print(f'init_read_w: \n' + f'{self.read_w}')
        # print(f'init_fb_w: \n' + f'{self.fb_w}')
        # print(f'init_ro: \n' + f'{self.ro}')
        # print(f'init_read: \n' + f'{self.read_o}')
        # print('\n\n\n')
        for cycle in range(10):
            # print(f'before_ru: ' + f'{self.ru}')
            # print(f'before_ro: ' + f'{self.ro}')
            # print(f'before_read: ' + f'{self.read_o}')
            self.ru_comp()
            # print(f'ru: ' + f'{self.ru}')
            self.ro_comp()
            # print(f'after_ro: ' + f'{self.ro}')
            self.read_o_comp()
        # input('instace cycle finish')
        # print(f'ru: {self.ru}')
        # print('\n')
        # print(f'ro: {self.ro}')
        # print('\n')
        # print(f'read_o: {self.read_o}')
        # print('\n\n')


    # noinspection PyMethodMayBeStatic
    # def implement_multi_tau(func):
    #     # noinspection PyUnresolvedReferences,PyStatementEffect,PyCallingNonCallable
    #     @functools.wraps(func)
    #     def deco(self):
    #         func(self)
    #         if len(self.v_tau) >= 2:
    #             n_tau = list(map(int, deepcopy(self.n_tau)))  # type: List[int]
    #             off_s = 0
    #             for len_n_tau in range(1, len(self.v_tau)):
    #                 leeking_rate = (1 / self.v_tau[len_n_tau])
    #                 bu_ru = (1 - leeking_rate) * self.old_ru + leeking_rate * self.bu_ru
    #                 n_tau[len_n_tau] += off_s
    #                 self.ru[off_s: n_tau[len_n_tau]] = bu_ru[off_s: n_tau[len_n_tau]]#todo tauごとの配列をそれぞれつくっておくべき、外部からの参照がしやすい
    #                 off_s = n_tau[len_n_tau]
    #         else:
    #             pass
    #     return deco

    # noinspection PyArgumentList
    # @jit(nopython=True)
    # @implement_multi_tau
    def ru_comp(self):
        # print(f'in_o: {self.in_o}')
        # print(f'in_w: {self.in_w}')
        # print('\n')
        self.old_ru = self.ru
        # print(f'old_ru: {self.old_ru}')
        # print('\n')
        # print(f'ro: {self.ro}')
        # print(f'rw: {self.rw}')
        # print('\n')
        # print(f'read_o: {self.read_o}')
        # print(f'fb_w: {self.fb_w}')
        # print('\n')

        self.new_ru = self.in_w.dot(self.in_o) + self.rw.dot(self.ro) + self.fb_w.dot(self.read_o)
        # self.bu_ru = self.ru
        self.leeking_rate = 1 / self.v_tau[0]
        self.ru = (1 - self.leeking_rate) * self.old_ru + self.leeking_rate * self.new_ru

        if len(self.v_tau) >= 2:
            n_tau = list(self.n_tau)  # type: List[int]
            self.off_s = 0
            for off_n_tau in range(1, len(self.v_tau)):
                self.leeking_rate = 1 / self.v_tau[off_n_tau]
                self.bu_ru = (1 - self.leeking_rate) * self.old_ru + self.leeking_rate * self.new_ru
                n_tau[off_n_tau] += self.off_s
                self.ru[self.off_s: n_tau[off_n_tau]] = self.bu_ru[self.off_s: n_tau[off_n_tau]]  # todo tauごとの配列をそれぞれつくっておくべき、外部からの参照がしやすい
                self.off_s = n_tau[off_n_tau]
        else:
            # print('inner else')
            # print('\n\n\n\n')

            pass
        # print(f'new_ru: {self.ru}')
        # print('\n')

    # @jit(nopython=True)
    def ro_comp(self):
        self.ro = xp.tanh(self.ru)

    # @jit(nopython=True)
    def read_o_comp(self):
        self.read_o = self.read_w.dot(self.ro)

    # @jit(nopython=True)
    def reset_net(self):
        # self.ru = xp.random.uniform(-0.1, 0.1, self.num)
        self.ru = xp.random.normal(0, 1.0, self.num)
        self.ro = xp.tanh(self.ru)
        self.read_o = xp.zeros(self.num_out)

    # @jit(nopython=True)
    # def pre_implement_multi_tau(func):
    #     # noinspection PyUnresolvedReferences,PyStatementEffect,PyCallingNonCallable
    #     @functools.wraps(func)
    #     def pre_deco(self, in_o):
    #         func(self, in_o)
    #         if len(self.v_tau) >= 2:
    #             n_tau = list(map(int, deepcopy(self.n_tau)))  # type: List[int]
    #             off_s = 0
    #             for len_n_tau in range(1, len(self.v_tau)):
    #                 leeking_rate = (1 / self.v_tau[len_n_tau])
    #                 bu_ru = (1 - leeking_rate) * self.pre_old_ru + leeking_rate * self.pre_bu_ru
    #                 n_tau[len_n_tau] += off_s
    #                 self.pre_ru[off_s: n_tau[len_n_tau]] = bu_ru[off_s: n_tau[len_n_tau]]
    #                 off_s = n_tau[len_n_tau]
    #         else:
    #             pass
    #     return pre_deco

    # @jit(nopython=True)
    # @pre_implement_multi_tau
    # def pre_ru_comp(self, in_o: xp.ndarray):
    #     self.pre_old_ru: xp.ndarray = self.ru
    #     self.pre_ru: xp.ndarray = self.in_w.dot(in_o) + self.rw.dot(self.ro) + self.fb_w.dot(self.read_o)
    #     self.pre_bu_ru: xp.ndarray = self.pre_ru
    #     leeking_rate = (1 / self.v_tau[0])
    #     self.pre_ru = (1 - leeking_rate) * self.pre_old_ru + leeking_rate * self.pre_bu_ru

    # @jit(nopython=True)
    # def pre_ro_comp(self):
    #     self.pre_ro = self.fn(self.pre_ru)


if __name__ == "__main__":
    xp.random.seed(1)
    EPISODE = 1
    # noinspection PyTypeChecker
    # rv = Reservoir(2, 10, 1, v_tau=(10, 2), n_tau=(4, 6))
    # rv = Reservoir(7, 1000, 10, v_tau=(10, 2), n_tau=(600, 400))
    rv = Reservoir(2, 10, 1, v_tau=(10, 2), n_tau=(3, 7))
    fg = fg.FG()
    rtp = rtp.RTP(Num_steps=EPISODE, Num_fig=4, figsize=(10, 5))
    fg.set_wave(Hz=1)
    rtp.init_sub_graph(0, "target")

    for ep in range(EPISODE + 1):
        # print(f'step: {ep}')
        # rv.ru_comp((rv.ltau, rv.stau), (rv.num / 2, rv.num / 2), rv_in=rv.in_o, rec_in=rv.old_ro, fb_in=rv.read_o)
        rv.in_o = xp.array([0.5, 0.2])
        rv.ru_comp()
        rv.ro_comp()
        # print(rv.ro[0:6])
        # print('\n\n')
        rv.read_o_comp()
        t = fg.gene_wave("wave1", ep)
        rtp.update_all([rv.read_o[0], rv.ro[0], rv.ro[1], rv.ro[2]])
        rtp.update_sub_graph(t, "target")
        if ep % 1000 == 0:
            rtp.plot()
    plt.show()
