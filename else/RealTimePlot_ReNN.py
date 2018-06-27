#!/usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random as rd

###############################################################

class RTP:
    def __init__(self, fig_number=10, Num_steps=100, Num_fig=3, y_low=-1.0, y_high=1.0, figsize=(20,10)):
#===="各パラメータ"=====
        self.Num_fig = Num_fig      # グラフの数
        self.Num_steps = Num_steps  # 最大ステップ数
        self.fig = plt.figure(num=fig_number, figsize=figsize)  # フィギュアサイズ
#===="リスト関係"=====
        self.frame_list = []        # ラインの外枠(位置)の入れ物
        self.main_lines = [0] * Num_fig  # メインラインプロット情報格納リスト
        self.sub_graph = {}         # サブライン(一つのfigに２つ以上波形を入れる時の追加分のライン)の情報を入れるディクショナリ
        self.sub_lines = []         # サブラインのプロット情報格納リスト
        self.x_list = range(Num_steps)                              # X軸の座標（範囲）
        self.y_list = [[0] * Num_steps for j in range(Num_fig)]     # Y軸の座標 #複数のラインがある場合のために多次元リストにしている
        #[[0.0] * Num_steps] * Num_fig]じゃダメ！！！！　要素に代入した時に値すべてが変わるから

#====="グラフの位置情報とプロット情報の格納"======
        for iii in range(Num_fig):# グラフの位置情報
            self.frame_list.append(self.fig.add_subplot(Num_fig, 1, iii+1))  #各グラフの位置(外枠)を格納
            self.frame_list[iii].set_ylim([y_low, y_high])                   #各グラフの振幅の範囲を決定

        for iii in range(Num_fig):# プロット情報
            self.main_lines[iii], = self.frame_list[iii].plot(self.x_list, self.y_list[iii])#plot情報(メインライン)を実際に格納
        plt.pause(.01)

##############################################################################################################################

# %%%%"figに単数のサブラインを入れるための関数"%%%%%
    def init_sub_graph(self, plot_num, fig_name):
        if plot_num >= self.Num_fig:#plot_numはサブラインを入れるfigの番号
            input('There is not such plot_number. Please refer the number of figures again')
        else:
            self.sub_graph[fig_name] = [plot_num, [0]*self.Num_steps, len(self.sub_lines)]#サブラインの情報を決めている[何番目のfigに入れるか、サブラインの値、サブラインのプロット数]
            self.sub_lines.append(0)    #sub_linesの初期化#これをしないとplotオブジェクトを代入できない。
            self.sub_lines[-1], = self.frame_list[plot_num].plot(self.x_list, self.sub_graph[fig_name][1])#plot情報(サブライン)を実際に格納

# %%%%"figに複数のサブラインを入れるための関数"%%%%%
    def init_sub_graphs(self, plot_num, fig_name):
        if len(plot_num) != len(fig_name):#plot_numとfig_nameはここではリスト型の引数
            input('arguments should be same length')
        else:
            for (num, name) in zip(plot_num, fig_name):
                self.sub_graph[name] = [num, [0]*self.Num_steps, len(self.sub_lines)]
                self.sub_lines.append(0)
                self.sub_lines[-1], = self.frame_list[num].plot(self.x_list, self.sub_graph[name][1])

# %%%%%"サブラインにプロット情報を入力（更新）する関数"%%%%%
    def update_sub_graph(self, val, name):#init_sub_graph(s)では要素プロットする要素すべてが0なのでここで値を入れてやる。
        self.sub_graph[name][1].pop(0)          #sub_graphの[0]*self.Num_stepsの最初の要素を除外し、
        self.sub_graph[name][1].append(val)     #そしてsub_graphの[0]*self.Num_stepsの最後の要素にvalを代入している
        self.sub_lines[self.sub_graph[name][2]].set_data(self.x_list, self.sub_graph[name][1])
        #設定したplot情報を更新する際にset_dataを使う

# %%%%%"メインラインにプロット情報を入力（更新）する関数"%%%%%
    def update(self, num, val):
        self.y_list[num].pop(0)
        self.y_list[num].append(val)
        self.main_lines[num].set_data(self.x_list, self.y_list[num])

# %%%%%"メインラインのすべて一挙に入力（更新）する関数"%%%%%
    def update_all(self, list_val):
        if len(list_val) != self.Num_fig:#list_valはメインラインの数
            print("not match length Num_fig and update_list")
        else:
            for iii in range(self.Num_fig):
                self.y_list[iii].pop(0)
                self.y_list[iii].append(list_val[iii])
                self.main_lines[iii].set_data(self.x_list, self.y_list[iii])
# %%%%%"プロット間の時間間隔の指定"%%%%%
    def plot(self):
        plt.pause(.01)

if __name__ == '__main__':
    rtp = RTP()
    rtp.init_sub_graph(0, "test")
    rtp.init_sub_graphs([1,2], ["A","B"])
    for iii in range(1000):
        rtp.update_all([rd.uniform(-1.0,1.0),rd.uniform(-1.0,1.0),rd.uniform(-1.0,1.0)])
        rtp.update_sub_graph(rd.uniform(-1.0,1.0), "test")
        if iii % 10 == 0:
            rtp.plot()


