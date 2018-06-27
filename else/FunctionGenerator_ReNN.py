#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import math
import random
# import time
# import re
# import sys
# import pickle
from collections import OrderedDict

import RealTimePlot_ReNN
# print(__name__)
#########################################################################
seed_val = 2
random.seed(seed_val)
np.random.seed(seed_val)

class FG:
#====="波形の情報を格納するディクショナリを作成"=====
    def __init__(self):
        self.Wave_inf = {}  #すべての波形の情報の入れ物

#====="波形のパラメータを設定＆Wave_inf{}へ情報を格納"=====(((wave:波形の種類(sin1とか) name:波形の任意の名前(wave1とか))))
    def set_wave(self, waveform = "sin1", name = "wave1", Hz = 2, dt = 0.001):
        data = [waveform, Hz, dt]       #一つの波形の情報
        self.Wave_inf[name] = data  #Wave_inf{}の中に波形の名前と対応させてその波形の情報を格納

#====="実際に波形を生み出す関数"=====
    def gene_wave(self, name, cont):#contは試行回数など
        return self.wave_function(self.Wave_inf[name][0], self.Wave_inf[name][1], self.Wave_inf[name][2]*cont)

#====="生み出す波形の関数を保管している関数"=====
    def wave_function(self, waveform, Hz, time):
        if waveform == "sin1":
            return math.sin(2*math.pi*Hz*time)
        elif waveform == "sin2":
            return math.sin(2*math.pi*Hz*time) * math.sin(2*math.pi*Hz*time)
        else:
            return 0.0

def main():
    rtp = RealTimePlot_ReNN.RTP(Num_steps = 1000, Num_fig = 1)
    fg = FG()   #FG()インスタンスをfgに代入する
    fg.set_wave(waveform = "sin2", name = "wave1", Hz = 2, dt = 0.001)#fgの中のset_wave()の設定を更新している
    for iii in range(10000):
        val = fg.gene_wave("wave1", iii)
        rtp.update(0, val)
        if iii % 5 == 0:
            rtp.plot()
    input("FINISH")

if __name__ == '__main__':
    main()


