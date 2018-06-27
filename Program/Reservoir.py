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

class Reservoir:
    _connect_rate = 10

    def __init__(self, num_in: int, num_neuron: int, num_fb: int, tau: int, p_connect: int = 10,
                 v_lambda: float = 1.8, in_wscale=1.0, fb_wscale=1.0):
        self._tau = None
        self._rw_scale = None

    @property
    def connect_rate(self):
        return self._connect_rate

    @connect_rate.setter
    def connect_rate(self, rate):
        self._connect_rate = rate

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau

    @property
    def rw_scale(self):
        return self._rw_scale

    @rw_scale.setter
    def rw_scale(self, scale):
        self._rw_scale = scale

    @property
    def in_w(self):






        self.num_in: int = num_in
        self.num: int = num
        self.num_out: int = num_out
        self.p_connect: int = p_connect
        self.inf_con = xp.array(xp.random.choice(2, (num, num),
                                                 p=[(100 - p_connect) / 100, p_connect / 100]))
        self.v_lambda: float = v_lambda
        self.v_tau: Tuple[float] = v_tau
        self.n_tau: Tuple[float] = n_tau

        self.in_w = xp.random.uniform(-in_wscale, in_wscale, (num, num_in))
        self.rw = v_lambda * xp.random.normal(0,
                                              1 / math.sqrt
                                              (float(num * p_connect / 100)), (num, num))
        self.rw *= self.inf_con
        self.read_w = xp.random.normal(0, 1 / math.sqrt(float(num)), (num_out, num))
        self.fb_w = xp.random.uniform(-fb_wscale, fb_wscale, (num, num_out))
        self.ru = xp.random.uniform(-1.0, 1.0, num)
        self.old_ru: xp.ndarray
        self.bu_ru: xp.ndarray
        self.in_o = xp.zeros(num_in)
        self.ro = xp.tanh(self.ru)
        self.pre_ro: xp.ndarray
        self.read_o = xp.zeros(num_out)
        for cycle in range(10):
            self.ru_comp()
            self.ro_comp()
            self.read_o_comp()

    def ru_comp(self):
        self.old_ru = self.ru

        self.new_ru = self.in_w.dot(self.in_o) + self.rw.dot(self.ro) + self.fb_w.dot(self.read_o)
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

    def ro_comp(self):
        self.ro = xp.tanh(self.ru)

    def read_o_comp(self):
        self.read_o = self.read_w.dot(self.ro)

    def reset_net(self):
        # self.ru = xp.random.uniform(-0.1, 0.1, self.num)
        self.ru = xp.random.normal(0, 1.0, self.num)
        self.ro = xp.tanh(self.ru)
        self.read_o = xp.zeros(self.num_out)


if __name__ == "__main__":
    # xp.random.seed(1)
    # EPISODE = 1
    # # noinspection PyTypeChecker
    # # rv = Reservoir(2, 10, 1, v_tau=(10, 2), n_tau=(4, 6))
    # # rv = Reservoir(7, 1000, 10, v_tau=(10, 2), n_tau=(600, 400))
    # rv = Reservoir(2, 10, 1, v_tau=(10, 2), n_tau=(3, 7))
    # fg = fg.FG()
    # rtp = rtp.RTP(Num_steps=EPISODE, Num_fig=4, figsize=(10, 5))
    # fg.set_wave(Hz=1)
    # rtp.init_sub_graph(0, "target")
    #
    # for ep in range(EPISODE + 1):
    #     # print(f'step: {ep}')
    #     # rv.ru_comp((rv.ltau, rv.stau), (rv.num / 2, rv.num / 2), rv_in=rv.in_o, rec_in=rv.old_ro, fb_in=rv.read_o)
    #     rv.in_o = xp.array([0.5, 0.2])
    #     rv.ru_comp()
    #     rv.ro_comp()
    #     # print(rv.ro[0:6])
    #     # print('\n\n')
    #     rv.read_o_comp()
    #     t = fg.gene_wave("wave1", ep)
    #     rtp.update_all([rv.read_o[0], rv.ro[0], rv.ro[1], rv.ro[2]])
    #     rtp.update_sub_graph(t, "target")
    #     if ep % 1000 == 0:
    #         rtp.plot()
    # plt.show()
