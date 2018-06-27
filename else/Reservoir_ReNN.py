#!/usr/bin/python3          #shebang(シバン)=>インタープリタの階層の指定
# -*- coding: utf-8 -*-     #文字エンコーディングの指定
from Cython.Shadow import profile

BACKPINK = '\033[105m'
BACKGREEN = '\033[104m'
BACKGREEN = '\033[102m'
TEXTLIGHTBLUE = '\033[96m'
TEXTPINK = '\033[95m'
TEXTBLUE = '\033[94m'
TEXTGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

import numpy as xp
# import cupy as xp
import matplotlib.pyplot as plt
from pylab import *
import random
import math

from time import perf_counter
import re
import os
import sys
from typing import Optional, Tuple, Callable

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import RealTimePlot_ReNN
import FunctionGenerator_ReNN

print(TEXTGREEN + 'START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' + ENDC)
#################################################################################
seed_val = 1
random.seed(seed_val)
xp.random.seed(seed_val)
np.random.seed(seed_val)


# noinspection PyTypeChecker
class Reservoir:
    def __init__(self, Num_Rneuron=1000, Per_connect=10, Num_out=1, Num_in=2, Val_lambda=1.8,
                 dt=1, Stau=2, Ltau=10, Per_Stau=40, in_w_scale=1.0, fb_w_scale=1.0):
        #######"ニューロン数"#######
        self.Num_input = Num_in  # Input部のニューロン数
        self.Num_Rneuron = Num_Rneuron  # Reservoir内のニューロン数
        self.Num_Rout = Num_out  # Reedout部のニューロン数
        #######"各係数"#######
        self.Per_connect = Per_connect  # 結合割合[%]
        self.Val_lambda = Val_lambda  # Reservoir内の結合係数のスケールλ
        self.dt = dt  # シミュレーションステップΔt
        self.Stau = Stau  # 小さい時定数Small_τ
        self.Ltau = Ltau  # 大きい時定数Large_τ
        self.Num_Stau = int(Per_Stau / 100 * Num_Rneuron)  # 時定数の小さいニューロンの導入数
        self.AAA = xp.ones(self.Num_Rneuron) * (1.0 - self.dt / self.Ltau)
        self.BBB = xp.ones(self.Num_Rneuron) * (self.dt / self.Ltau)  # leekrate
        self.CCC = xp.ones(self.Num_Rneuron) * Val_lambda
        self.DDD = xp.ones(self.Num_Rneuron) * (1.0 - self.dt / self.Stau)
        self.EEE = xp.ones(self.Num_Rneuron) * (self.dt / self.Stau)
        self.connect_inf = xp.array(
            xp.random.choice(2, (Num_Rneuron, Num_Rneuron), p=[(100 - Per_connect) / 100.0, Per_connect / 100.0]))
        #######"内部状態"#######
        self.rec_u = xp.random.uniform(-1.0, 1.0, self.Num_Rneuron)  # Reservoir内ニューロンの内部状態(計算に使う) # (1000,)
        self.rec_u_s = xp.random.uniform(-1.0, 1.0, self.Num_Rneuron)  # Reservoir内ニューロンの内部状態(時定数小さい)
        #######"重み値"#######
        self.W_input = xp.random.uniform(-in_w_scale, in_w_scale, (Num_in, Num_Rneuron))  # (2,1000)
        self.W_Rec = Val_lambda * xp.random.normal(0, 1 / math.sqrt(float(Num_Rneuron * Per_connect / 100.0)),
                                                   (Num_Rneuron, Num_Rneuron))  # (1000, 1000)
        self.W_Rec *= self.connect_inf
        self.W_Rout = xp.random.normal(0, 1 / math.sqrt(float(Num_Rneuron)), (Num_Rneuron, Num_out))
        self.W_Fback = xp.random.uniform(-fb_w_scale, fb_w_scale, (Num_out, Num_Rneuron))
        #######"出力"#######
        self.net_in = xp.zeros(Num_in)  # Input部からの入力 # (2,)
        self.old_rec_o = xp.tanh(self.rec_u)  # Reservoir内の出力のバックアップ # (1000,)
        self.rec_o = xp.tanh(self.rec_u)  # Reservoir内の出力 # (1000,)
        self.old_O_Rout = xp.zeros(Num_out)  # Readout部の出力のバックアップ
        self.read_o = xp.zeros(Num_out)  # Readout部の出力
        for iii in range(10):
            self.rec_u_comp(net_in=self.net_in)
            self.rec_o_comp()
            self.Rout_comp()
            #########################################################################################################
            # %%%%%"Reservoir内ニューロンの内部状態の計算"%%%%%


    def rec_u_comp(self, net_in: ndarray, prediction: bool = False) -> Optional[float]:
        rec_u = (self.AAA * self.rec_u +
                 self.BBB * (
                     xp.dot(self.rec_o, self.W_Rec)  # (1000.)
                     + xp.dot(net_in, self.W_input)
                     + xp.dot(self.read_o, self.W_Fback))
                 )  # 時定数が大きいニューロンだけのリザバ
        rec_u_s = (self.DDD * self.rec_u +
                   self.EEE * (
                       xp.dot(self.rec_o, self.W_Rec)
                       + xp.dot(net_in, self.W_input)
                       + xp.dot(self.read_o, self.W_Fback))
                   )  # 時定数が小さいニューロンだけのリザバ
        rec_u[:self.Num_Stau] = rec_u_s[:self.Num_Stau]  # ここで時定数小さいニューロンが導入された。
        if prediction:
            return rec_u
        else:
            self.rec_u = rec_u

    # %%%%%"リザバの出力の計算"%%%%%
    m_rec_u_comp = Optional[Callable[[ndarray, bool], Optional[ndarray]]]

    def rec_o_comp(self, v_rec_u: m_rec_u_comp=None, prediction: bool=False) -> Optional[ndarray]:
        if prediction:
            return xp.tanh(v_rec_u)
        else:
            self.old_rec_o = self.rec_o
            self.rec_o = xp.tanh(self.rec_u)


    def one_cicle(self, net_in: Tuple[float], variable: bool=False, prediction: bool=False) -> Optional[Variable]:
        self.net_in[:] = net_in
        if prediction:
            # rec_u =
            rec_o = self.rec_o_comp(v_rec_u=self.rec_u_comp(net_in=self.net_in, prediction=True), prediction=True)
            if variable:
                return Variable(np.array([rec_o], dtype=np.float32))
            else:
                return rec_o
        else:
            self.rec_u_comp(net_in=self.net_in)
            self.rec_o_comp()
            if variable:
                return Variable(np.array([self.rec_o], dtype=np.float32))

            # def U_Rec_comp_ex(self, in_vals):
            #     self.U_Rec_ex_L = (self.AAA * self.rec_u +
            #                        self.BBB * (
            #                            xp.dot(self.rec_o, self.W_Rec)
            #                            + xp.dot(in_vals, self.W_input)
            #                            + xp.dot(self.read_o, self.W_Fback))
            #                        )  # 時定数が大きいニューロンだけのリザバ
            #
            #     self.U_Rec_ex_S = (self.DDD * self.rec_u +
            #                        self.EEE * (
            #                            xp.dot(self.rec_o, self.W_Rec)
            #                            + xp.dot(in_vals, self.W_input)
            #                            + xp.dot(self.read_o, self.W_Fback))
            #                        )  # 時定数が小さいニューロンを導入
            #
            #     self.U_Rec_ex_L[:self.Num_Stau] = self.U_Rec_ex_S[:self.Num_Stau]
            #
            #     return self.U_Rec_ex_L
            # return (self.AAA * self.rec_u +
            #               self.BBB * (
            #                   xp.dot(self.rec_o, self.W_Rec)
            #                   + xp.dot(in_vals, self.W_input)
            #                   + xp.dot(self.read_o, self.W_Fback))
            #               )  # 時定数が大きいニューロンだけのリザバ
    def O_Rec_comp_ex(self, o_vals):  # リザバ出力
        return xp.tanh(o_vals)

    # %%%%%"リードアウトの線形和の計算"%%%%%　
    def Rout_comp(self):  # リードアウト出力の予測値(線形和)
        self.old_O_Rout = self.read_o
        self.read_o = xp.dot(self.rec_o, self.W_Rout)

    # %%%%%"エージェントがゴールした後にネットワークの情報を一旦リセットする関数"%%%%%
    # これをしないとゴールした後、次に試行に移る際に一試行前の入力情報も入ってしまう(リザバは前の情報を保持しているから)から必要！！！！！！！！！！！
    def reset_network(self):
        self.rec_u = xp.random.uniform(-0.1, 0.1, self.Num_Rneuron)
        self.rec_o = xp.tanh(self.rec_u)
        self.Rout_val = xp.zeros(self.Num_Rout)


def main():
    net = Reservoir()  # netにReservoir_ReNN.Reservoir()オブジェクトを代入
    rtp = RealTimePlot_ReNN.RTP(Num_steps=10000, Num_fig=4, figsize=(10, 5))  # rtpにRealTimePlot_ReNN.RTP()オブジェクトを代入
    fg = FunctionGenerator_ReNN.FG()  # fgにFunctionGenerator_ReNN.FG()オブジェクトを代入

    fg.set_wave(waveform="sin1", name="wave1", Hz=1, dt=0.001)
    rtp.init_sub_graph(0, "target")
    for iii in range(10001):
        net.rec_u_comp()
        net.rec_o_comp()
        net.Rout_comp()
        target_val = fg.gene_wave("wave1", iii)
        rtp.update_all([net.read_o[0], net.rec_o[5], net.rec_o[6], net.rec_o[7]])
        # rtp.update_all([net.read_o[0], net.read_o[], net.read_o[0], net.read_o[0]])
        rtp.update_sub_graph(target_val, "target")
        # net.one_cicle((6.0,), variable=True)

        if iii % 1000 == 0:
            rtp.plot()
    input(TEXTGREEN + "FINISH" + ENDC)


if __name__ == '__main__':
    main()
