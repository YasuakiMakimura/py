#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import *
import random
from time import perf_counter
import re
import sys
import pickle
seed_val = 1
random.seed(seed_val)
np.random.seed(seed_val)

display_pace = .01
class Log:
    def __init__(self, max_ep, fig_number=0, origin=False, v_origin=(None, None), c_origin=(None, None)):
        self.fig = plt.figure(num = fig_number, figsize=(5,3))
        self.frame = self.fig.add_subplot(111)
        self.max_ep = max_ep
        self.step_log_data = [0]*1         # ステップごとに記録をとるリスト
        self.epi_log_data = []          # エピソードごとに記録をとるリスト
        self.fig_number = fig_number   # フィギュアナンバー
        self.log_lines = [0]*1
        self.org_lines = [0]*len(v_origin)#org:origin
        if origin:
            for n_ori in range(len(v_origin)):
                self.org_lines[n_ori], = self.frame.plot(range(self.max_ep), [v_origin[n_ori]]*self.max_ep, c=c_origin[n_ori])
            # self.org_lines[1], = self.frame.plot(range(self.max_ep), [0.8]*self.max_ep, c="red")



# %%%%%"ステップごとに記録を消去する関数"%%%%% use
    def delete_step(self):
        self.step_log_data = []

# %%%%%"記録リストの初期化のための関数"%%%%%
    def delete_all(self):
        self.step_log_data = []         # ステップごとに記録をとるリスト
        self.epi_log_data = []          # エピソードごとに記録をとるリスト

# %%%%%"ステップごとに記録リストに値を追加する関数"%%%%%
    def log_step(self, current_data):
        self.step_log_data.append(current_data)

# %%%%%"ステップごとの記録リストにデータのリストを追加する関数"%%%%% use
    def log_step_list(self, data_list):
        self.step_log_data.extend(data_list)

    def log_step_list2(self, data_list):
        self.step_log_data = []
        self.step_log_data.extend(data_list)

# %%%%%"ステップごとの記録リストをエピソードごとの記憶リストに追加する関数"%%%%% use
    def log_episode(self):
        self.epi_log_data.append(self.step_log_data)
        self.delete_step()

# %%%%%"出来上がった記録リストからグラフをプロットする関数"%%%%% use
    def plot_log(self, Num_epi=False, Num_steps=50, save_im=False, set_xrange=False, y_low=0, y_high=180,
                 grid_width=False, x_label="x", y_label="y",
                 fig_size=(18,12), fig_name="result.png", stop=False, fig_color="random"):
    # 最大ステップ数を代入
        step_max = self.max_step()
        set_step_max = Num_steps
    # episode数を変数Num_epiに代入
        if Num_epi == False:
            Num_epi = range(self.count_epi())# count_epi:エピソードの数を返す関数
    # フィギュアを準備
        plt.figure(self.fig_number, fig_size)# グラフナンバー、サイズ

        if set_xrange:
        # グラフの表示範囲指定
            plt.axis([0, set_step_max, y_low, y_high])

        else:
            plt.axis([0, step_max, y_low, y_high])
    # グラフの各設定
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
    # グラフの細かい設定ができるrcParamsを使っている# 線の間隔を指定
        mpl.rcParams["lines.linewidth"] = 1

        if grid_width != False:# 何か特別にX,Y軸の範囲、文字間隔を設定する場合
        # 引数grid_widthに二次元のリストを入れる！！！　# X, Y軸のグリッド線を入れる間隔などを指定
            plt.xticks(np.arange(0, step_max, grid_width[0]))
            plt.yticks(np.arange(y_low, y_high, grid_width[1]))
        self.log_lines = [0]*len((Num_epi))

        if fig_color == "random":
            for (iii, jjj) in zip(Num_epi, range(len(Num_epi))):
                self.log_lines[jjj], = plt.plot(np.arange(len(self.epi_log_data[iii][:])), np.array(self.epi_log_data[iii][:]))
        else:
            for (iii, jjj) in zip(Num_epi, range(len(Num_epi))):
                self.log_lines[jjj], = plt.plot(np.arange(len(self.epi_log_data[iii][:])), np.array(self.epi_log_data[iii][:]), c = fig_color)
                # self.delete_all()
        plt.tight_layout()
        if save_im:
            plt.savefig(fig_name)
        plt.pause(display_pace)
        if stop:
            key = input("Press any button to finish")

    def plot_log2(self, Num_epi=False, Num_steps=50, save_im=False, set_xrange=False, y_low=0, y_high=180,
                 grid_width=False, x_label="x", y_label="y",
                 fig_size=(18,12), fig_name="result.png", stop=False, fig_color="random"):
    # 最大ステップ数を代入
        step_max = self.max_step()
        set_step_max = Num_steps
    # episode数を変数Num_epiに代入
        if Num_epi == False:
            Num_epi = range(self.count_epi())# count_epi:エピソードの数を返す関数
    # フィギュアを準備
        plt.figure(self.fig_number, fig_size)# グラフナンバー、サイズ

        if set_xrange:
        # グラフの表示範囲指定
            plt.axis([0, set_step_max, y_low, y_high])

        else:
            plt.axis([0, step_max, y_low, y_high])
    # グラフの各設定
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
    # グラフの細かい設定ができるrcParamsを使っている# 線の間隔を指定
        mpl.rcParams["lines.linewidth"] = 1

        if grid_width != False:# 何か特別にX,Y軸の範囲、文字間隔を設定する場合
        # 引数grid_widthに二次元のリストを入れる！！！　# X, Y軸のグリッド線を入れる間隔などを指定
            plt.xticks(np.arange(0, step_max, grid_width[0]))
            plt.yticks(np.arange(y_low, y_high, grid_width[1]))
        self.log_lines = [0]*len((Num_epi))

        if fig_color == "random":
            for (iii, jjj) in zip(Num_epi, range(len(Num_epi))):
                self.log_lines[jjj], = plt.plot(np.arange(len(self.step_log_data[iii][:])), np.array(self.step_log_data[iii][:]))
        else:
            for (iii, jjj) in zip(Num_epi, range(len(Num_epi))):
                self.log_lines[jjj], = plt.plot(np.arange(len(self.step_log_data[iii][:])), np.array(self.step_log_data[iii][:]), c = fig_color)
                # self.delete_all()
        plt.tight_layout()
        if save_im:
            plt.savefig(fig_name)
        plt.pause(display_pace)
        if stop:
            key = input("Press any button to finish")


# %%%%%"実行結果のグラフを名前を付けて保存するときの関数"%%%%%
    def save(self, fig_name=""):
        plt.figure(self.fig_number)
        plt.savefig(fig_name)

# %%%%%"プロット情報を削除"%%%%%
    def remove_log(self):
        plt.figure(self.fig_number)
        self.log_lines.remove()

# %%%%%""%%%%%
    def replot_log(self, Num_epi=False, save_im=False, stop=False, fig_name="result.png"):
        plt.figure(self.fig_number)
        if Num_epi == False:
        # episode数を変数Num_epiに代入
            Num_epi = range(self.count_epi())# count_epi:エピソードの数を返す関数

        for iii in range(len(self.log_lines)):
            self.log_lines[iii].remove()
        self.log_lines = [0]*(len(Num_epi))
        for (iii,jjj) in zip(Num_epi,range(len(Num_epi))):
            self.log_lines[jjj], = plt.plot(np.arange(len(self.epi_log_data[iii][:])), np.array(self.epi_log_data[iii][:]))
        if save_im:
            plt.savefig(fig_name)
        plt.pause(display_pace)
        if stop == True:
            key = input('Press any button to finish')

    def replot_log2(self):
        # self.log_lines[0], = plt.plot(range(len(self.step_log_data[:])), self.step_log_data[:])
        self.log_lines[0].set_data(range(len(self.step_log_data[:])), self.step_log_data[:])
        self.delete_step()

    def replot_log3(self):
        self.lines = [0]
        self.lines[0], = plt.plot(range(len(self.step_log_data[:])), self.step_log_data[:])
        self.delete_step()

# %%%%%"各エピソードが何ステップあったかをリストにして返す関数"%%%%% use
    def count_data(self):
        Num_index = []
        for iii in self.epi_log_data:
            Num_index.append(len(iii))
    # epi_log_dataのそれぞれのオフセットのlenをとっているのでNum_indexにはそれぞれのepisodeのstep数が入る
    #     Num_index = [len(iii) for iii in self.epi_log_data]
        return Num_index

# %%%%%"エピソードの数を返す関数"%%%%% use
    def count_epi(self):
        return len(self.count_data())# count_data:各エピソードが何ステップあったかをリストにして返す関数

# %%%%%"全エピソードの中で最大ステップがかかった時のステップ数を返す関数"%%%%% use
    def max_step(self):
        return max(self.count_data())

    def replot_log_color(self, col_list, Num_epi=False, save_im=False, stop=False, fig_name='result.png', col_max=1.0,
                         col_min=0.0):
        plt.figure(self.fig_number)
        for iii in range(len(self.log_lines)):
            self.log_lines[iii].remove()
        self.log_lines = [0] * (len(Num_epi))
        for (iii, jjj) in zip(Num_epi, range(len(Num_epi))):
            color_value = (col_list[jjj] - col_min) / (col_max - col_min)
            self.log_lines[jjj], = plt.plot(np.arange(len(self.ep_log_data[iii][:])),
                                            np.array(self.ep_log_data[iii][:]),
                                            c=cm.hsv((1.0 - color_value) * (2.0 / 3.0)))
        if save_im:
            plt.savefig(fig_name)
        plt.pause(display_pace)
        if stop == True:
            key = input('Press any button to finish')

    def save_log_data(self, name='log_data.pickle'):  # ログデータの保存
        log_data = open(name, 'w')
        pickle.dump(self.epi_log_data, log_data)
        log_data.close()

    def load_log_data(self, name='log_data.pickle'):  # ログデータの読み込み
        log_data = open(name, 'r')
        self.epi_log_data = pickle.load(log_data)
        log_data.close()
