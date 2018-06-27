#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
# import cupy as cp
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib.pyplot as plt
from pylab import *
import math
from typing import Union, Optional

from time import perf_counter
import re
import sys
import os
import argparse

import RealTimePlot_ReNN
import random
# import Parameter_ReNN
print(__name__)
##############################################################
# seed_val = 1
# random.seed(seed_val)
# np.random.seed(seed_val)
# Param = Parameter_ReNN.Database()

# Num_out=db.p['out_top']# 3       # フィードバックされてくるユニットの出力数
# Num_1=db.p['out_l1']# 100           # 階層型NNの第一中間層ニューロン数
# Num_2=db.p['out_l2']# 40           # 階層型NNの第二中間層ニューロン数
# Num_3=db.p['out_l3']# 10           # 階層型NNの第三中間層ニューロン数
# Num_in=db.p['n_h_neuron']      # リザバから階層型NNへの入力数
# Num_direct_in=db.p['n_in'] # ネットワークの入力数

#####"ネットワークの構築"#####
class MyChain(Chain):
    def __init__(self, n_in, n_bypass, n_l1, n_l2, n_l3, n_top, nor_w=(False, (None, None, None, None)), uni_w=(False, (None, None, None, None)), uni_bypass_w=(False, None), nobias=(False, False, False, False, False)):
        initializer_out = []
        if all([nor_w[0], uni_w[0]]):
            raise Exception(f'Both of arg(nor_w) and arg(uni_w) in the same time can not be set')

        if uni_w[0]:
            for iii in range(len(uni_w[1])):
                initializer_out.append(initializers.Uniform(scale=uni_w[1][iii]))
        elif nor_w[0]:
            for iii in range(len(nor_w[1])):
                initializer_out.append(initializers.LeCunNormal(scale=nor_w[1][iii]))
        else:
            initializer_out = (None, None, None, None)

        if uni_bypass_w[0]:
            initializer_bypass = initializers.Uniform(scale=uni_bypass_w[1])
        else:
            initializer_bypass = None

        super(MyChain, self).__init__(
            net_in = L.Linear(n_bypass, n_l1, nobias=nobias[0], initialW=initializer_bypass),
            l1 = L.Linear(n_in, n_l1, nobias=nobias[1], initialW=initializer_out[0]),
            l2 = L.Linear(n_l1, n_l2, nobias=nobias[2], initialW=initializer_out[1]),
            l3 = L.Linear(n_l2, n_l3, nobias=nobias[3], initialW=initializer_out[2]),
            lout = L.Linear(n_l3, n_top, nobias=nobias[4], initialW=initializer_out[3]),

            # 出力ユニットが普通のニューロン一個のやつ
            ldirect_r = L.Linear(n_in, n_top, nobias=False),
            ldirect_s = L.Linear(n_bypass, n_top, nobias=False),# 入力からのバイパス
        )

# %%%%%"階層型NNの各層での出力を求める関数(戻り値は各層の出力)"%%%%%
    def ff_comp(self, net_in, bypass, net_type=1):# net_in:リザバから階層型NNへの入力、　bypass:入力層からのバイパス
        if net_type == 1:
            h1 = F.tanh(self.l1(net_in)+self.net_in(bypass))
            h2 = F.tanh(self.l2(h1))
            h3 = F.tanh(self.l3(h2))
            y = F.tanh(self.lout(h3))
            return h1, h2, h3, y

        elif net_type == 2:
             h1 = F.tanh(self.l1(net_in))
             h2 = F.tanh(self.l2(h1))
             h3 = F.tanh(self.l3(h2))
             y = F.tanh(self.lout(h3))
             return h1, h2, h3, y

        elif net_type == 3:
            h1 = F.tanh(self.net_in(bypass))
            h2 = F.tanh(self.l2(h1))
            h3 = F.tanh(self.l3(h2))
            y = F.tanh(self.lout(h3))
            return h1, h2, h3, y

        elif net_type == 4:
            y = F.tanh(self.ldirect_r(net_in)+self.ldirect_s(bypass))
            return y, y, y, y

    def send_w(self):
        return (self.l1.W.data[:], self.l2.W.data[:], self.l3.W.data[:], self.lout.W.data[:], self.net_in.W.data[:],
                self.l1.b.data[:], self.l2.b.data[:], self.l3.b.data[:], self.lout.b.data[:], self.net_in.b.data[:])
        # .dataにより、Variableクラスからarrayに取り出す

    def receive_w(self, w_val):
        self.l1.W.data[:] = w_val[0]
        self.l2.W.data[:] = w_val[1]
        self.l3.W.data[:] = w_val[2]
        self.lout.W.data[:] = w_val[3]
        self.net_in.W.data[:] = w_val[4]
        self.l1.b.data[:] = w_val[5]
        self.l2.b.data[:] = w_val[6]
        self.l3.b.data[:] = w_val[7]
        self.lout.b.data[:] = w_val[8]
        self.net_in.b.data[:] = w_val[9]

def main():
    model = MyChain() # ネットワークのインスタンス化
    optimizer = optimizers.Adam() # Optimizerを作成(インスタンス化):手法はAdam
    optimizer.use_cleargrads() # 計算の効率化のために入れる(cleargradsを使えるようにしている)# 重みを一度に初期化できる
    optimizer.setup(model) # Optimizerにモデルをセット


if __name__ == '__main__':
    main()

