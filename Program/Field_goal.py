#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import math
import random
from typing import Tuple, Union, Optional, Iterable
from time import perf_counter
######################################################################
# seed_val = 1
# random.seed(seed_val)
# np.random.seed(seed_val)

display_pace = 0.01
print(__name__)

class Field:
    def __init__(self, field_x=10.0, field_y=10.0, agent_x=10.0, agent_y=1.0,
                 Num_plot=10, fig_number=0,
                 rew=0.2, pena=-0.1, spena=-1.0, sw_scale=10.0, testing=False):
#====="フィールド自体のパラメータ"=====#todo 初期位置がランダムかどうかをコンストラクタの引数で指定してその引数次第で__dict__の中身の関数を書き換える＝＞グローバル関数（orクラス内の関数)を無理やり__dict__(or__class__)に入れる。それによって引数で初期位置がランダムかどうかを指定できる
        self.field_x = field_x          # フィールドの横幅
        self.field_y = field_y          # フィールドの縦幅
#====="報酬と罰"=====
        self.penalty_wall = pena    # 壁の罰
        self.spena = spena
        self.reward_goal = rew      # ゴールの報酬
#====="ゴールのパラメータ"=====
        self.goal_rl_flag = 0       # ゴールがどっちかのフラグ
        self.x_goal1 = field_x*4.0/5.0  # ゴール1の位置X座標     ?ゴール位置の候補
        self.y_goal1 = field_y*4.0/5.0  # ゴール1の位置Y座標
        self.x_goal2 = field_x*1.0/5.0  # ゴール2の位置X座標
        self.y_goal2 = field_y*4.0/5.0  # ゴール2の位置Y座標
        self.x_goal = self.x_goal1  # ゴールの位置X座標
        self.y_goal = self.y_goal1  # ゴールの位置Y座標
        self.radius_goal = field_x/10.0      # ゴールの半径
        self.cir_goal1 = plt.Circle((self.x_goal1, self.y_goal1), self.radius_goal, fc="#000077")    # ゴール1を描いている
        self.cir_goal2 = plt.Circle((self.x_goal2, self.y_goal2), self.radius_goal, fc="#000077")    # ゴール2を描いている
#====="スイッチのパラメータ"=====
        self.switch_push_flag = 0                # 一回でもスイッチを押したかのフラグ
        self.x_switch = field_x/2.0              # スイッチのX座標
        self.y_switch = field_y/3.0              # スイッチのY座標
        self.radius_switch = field_x/10.0         # スイッチの半径
        self.switch_on = 0                       # スイッチオンオフフラグ
        self.sw_scale = sw_scale
        self.cir_switch = plt.Circle((self.x_switch, self.y_switch), self.radius_switch, fc = "#007700")
#====="エージェントパラメータ"=====
        self.init_x_agent = field_x*1.0/5.0     # エージェント初期X座標
        self.init_y_agent = field_y*1.0/5.0     # エージェント初期Y座標
        self.x_agent = agent_x                  # エージェントの位置X座標
        self.y_agent = agent_y                  # エージェントの位置Y座標
        self.last_x_agent = agent_x             # 一回前のエージェントの位置X座標を入れるための変数
        self.last_y_agent = agent_y             # 一回前のエージェントの位置Y座標を入れるための変数
        self.color_agent = "red"            # エージェントの色

        self.Num_plot = Num_plot
        # self.traj_x = [self.init_x_agent]   # エージェントの軌道X座標の情報の格納リスト(単数)
        # self.traj_y = [self.init_y_agent]   # エージェントの軌道Y座標の情報の格納リスト(単数)
        self.traj_x = []   # エージェントの軌道X座標の情報の格納リスト(単数)
        self.traj_y = []   # エージェントの軌道Y座標の情報の格納リスト(単数)
        self.trajs_x = [[self.init_x_agent] for iii in range(Num_plot)]     # エージェントのゴールするまでの軌道(X座標)を格納するリスト
        self.trajs_y = [[self.init_y_agent] for iii in range(Num_plot)]     # エージェントのゴールするまでの軌道(Y座標)を格納するリスト
        self.trajectries = [0]*Num_plot
        self.ini_trajectries = [0]*Num_plot
        self.test_trajectries = [0]*Num_plot
#====="プロット関係のパラメータ"=====
        self.fig = plt.figure(num = fig_number, figsize = (6,6))
        self.frame = self.fig.add_subplot(111)
        axis([-1, self.field_x+1, -1, self.field_y+1])
        xlabel("X", fontsize=15)    # x軸ラベル
        ylabel("Y", fontsize=15)    # y軸ラベル
        xticks(fontsize=15)
        yticks(fontsize=15)
        # title("Simulation") # グラフのタイトル
        legend()            # グラフの凡例
        grid(True)          # グラフのグリッド線の表示

        self.frame.add_patch(self.cir_goal1)
        # self.frame.add_patch(self.cir_goal2)
        self.frame.add_patch(self.cir_switch)
        if not testing:
            for iii in range(Num_plot):
                self.trajectries[iii], = self.frame.plot(self.trajs_x[iii], self.trajs_y[iii], ms=5., marker='o', c="green")
                self.ini_trajectries[iii], = self.frame.plot(self.trajs_x[iii], self.trajs_y[iii], ms=10., marker='o', c="red")
        pause(display_pace)
#############################################################################################################
# %%%%%"フィールドのサイズを変更する関数"%%%%%
    def redefinition_field(self, f_x, f_y):
        self.field_x = f_x#フィールドの横幅
        self.field_y = f_y#フィールドの縦幅
        self.x_goal1 = f_x*4.0/5.0#ゴールの位置X座標
        self.y_goal1 = f_y*4.0/5.0#ゴールの位置Y座標
        self.x_goal2 = f_x*1.0/5.0#ゴールの位置X座標
        self.y_goal2 = f_y*4.0/5.0#ゴールの位置Y座標
        self.x_goal = self.x_goal1#ゴールの位置X座標
        self.y_goal = self.y_goal1#ゴールの位置Y座標
        self.x_switch = f_x/2.0
        self.y_switch = f_y/3.0
        self.radius_goal = f_x/10.0#ゴールの半径
        self.radius_switch = f_x/10.0#ゴールの半径
        axis([-1, self.field_x+1, -1, self.field_y+1])#グラフの表示範囲指定

# %%%%%"エージェントの座標を移動させる関数"%%%%%
    def move_agent(self, x_val, y_val):
        self.last_x_agent = self.x_agent
        self.last_y_agent = self.y_agent
        self.x_agent = self.x_agent + x_val
        self.y_agent = self.y_agent + y_val

# %%%%%"今の座標をリストに入れるための関数"%%%%% #ただし最初にこの関数を呼び出した時はdefaultで初期位置が格納されている
    def add_p_log(self, posi_x: float, posi_y: float, init=False):
        self.traj_x.append(posi_x)
        self.traj_y.append(posi_y)
        if init:
            self.traj_x.append(posi_x)
            self.traj_y.append(posi_y)
            self.ini_trajectries[0].set_data(self.traj_x[:], self.traj_y[:])

    def add_p_log2(self, posi_x: float, posi_y: float):
        self.traj_x.append(posi_x)
        self.traj_y.append(posi_y)

    def ini_add_p_log(self, posi_x: float, posi_y: float):
        self.traj_x.append(posi_x)
        self.traj_y.append(posi_y)
        self.ini_trajectries[0].set_data(self.traj_x[:], self.traj_y[:])

    def ini_add_p_log_test(self, posi_x: float, posi_y: float, var_ep: int,
                           color=('#FF6347', '#008B8B', '#8B008B', '#FF1493', '#008000'),
                           ms=10):
        self.traj_x.append(posi_x)
        self.traj_y.append(posi_y)
        self.ini_trajectries[var_ep], = self.frame.plot(self.traj_x[:], self.traj_y[:],
                                                        ms=ms, marker='o',
                                                        c=color[var_ep])

    # %%%%%"ゴールするまでの座標をリストに入れるための関数"%%%%%
    def epi_log(self):
        self.trajs_x.append(self.traj_x)
        self.trajs_y.append(self.traj_y)
        self.traj_x = []
        self.traj_y = []
# %%%%%"座標情報をプロット情報としてリストに格納する関数"%%%%%
    def replot(self):
        for iii in reversed(range(self.Num_plot)):
            self.trajectries[self.Num_plot-1-iii].set_data(self.trajs_x[-1-iii], self.trajs_y[-1-iii])
        pause(display_pace)

    def replot2(self):
        self.trajectries[0].set_data(self.traj_x[:], self.traj_y[:])
        self.traj_x = []
        self.traj_y = []
        pause(display_pace)

    def replot_test(self, var_ep: int,
                    color=('#FF6347', '#008B8B', '#8B008B', '#FF1493', '#008000'), lw=1, ms=5):
        self.trajectries[var_ep], = self.frame.plot(self.traj_x[:], self.traj_y[:],
                                                    ms=ms, marker='o', c=color[var_ep], lw=lw)
        self.traj_x = []
        self.traj_y = []
        pause(display_pace)

# %%%%%"ゴールの図形を一旦消して、再度図形をプロットする関数"%%%%%
    def replot_goal(self):
        self.cir_goal1.remove()
        # self.cir_goal2.remove()
        self.cir_switch.remove()
        self.cir_goal1 = plt.Circle((self.x_goal1, self.y_goal1), self.radius_goal, fc="#000077")
        # self.cir_goal2 = plt.Circle((self.x_goal2, self.y_goal2), self.radius_goal, fc="#770000")
        self.cir_switch = plt.Circle((self.x_switch, self.y_switch), self.radius_switch, fc="#770000")
        self.frame.add_patch(self.cir_goal1)
        # self.frame.add_patch(self.cir_goal2)
        self.frame.add_patch(self.cir_switch)
        pause(display_pace)

# %%%%%"ゴールの中心との距離とスイッチの中心との距離をを計算する関数(報酬の判定に使用する)"%%%%%
    def objects_distance(self):
        dist_g1 = math.sqrt((self.x_agent - self.x_goal1)**2 + (self.y_agent - self.y_goal1)**2)
        dist_g2 = math.sqrt((self.x_agent - self.x_goal2)**2 + (self.y_agent - self.y_goal2)**2)
        dist_s = math.sqrt((self.x_agent - self.x_switch)**2 + (self.y_agent - self.y_switch)**2)
        return dist_g1, dist_g2, dist_s
# %%%%%"ゴールを踏まずにちゃんと最初にスイッチに来たか着てないかを判別する関数"%%%%%
    def switch_check(self):
    # ゴールとの距離、スイッチとの距離をそれぞれの変数に代入
        _1, _2, dist_s = self.objects_distance()# objects_distace:ゴールの中心との距離とスイッチの中心との距離をを計算する関数(報酬の判定に使用する)
        if dist_s <= self.radius_switch and self.goal_rl_flag == 0:# ちゃんと最初にスイッチに来た
            return self.sw_scale, 0.0
        elif dist_s <= self.radius_switch and self.goal_rl_flag == 1:# ゴールしてからスイッチに来た
            return 0.0, self.sw_scale
        return 0.0, 0.0
# %%%%%"軌道の結果をセーブする関数"%%%%%
    def save(self, fig_name="field.png"):
        self.fig.savefig(fig_name)
# %%%%%"目的地との相対角度を計算するための関数"%%%%%
    def objects_angle(self):
        # todo 0.0000000001
        dist_g1, dist_g2, dist_s = self.objects_distance()
        # sin_g1 = (self.y_goal1 - self.y_agent)/(dist_g1+0.0000000001)
        # cos_g1 = (self.x_goal1 - self.x_agent)/(dist_g1+0.0000000001)
        # sin_g2 = (self.y_goal2 - self.y_agent)/(dist_g2+0.0000000001)
        # cos_g2 = (self.x_goal2 - self.x_agent)/(dist_g2+0.0000000001)
        # sin_sw = (self.y_switch - self.y_agent)/(dist_s+0.0000000001)
        # cos_sw = (self.x_switch - self.x_agent)/(dist_s+0.0000000001)
        sin_g1 = (self.y_goal1 - self.y_agent)/(dist_g1+1)
        cos_g1 = (self.x_goal1 - self.x_agent)/(dist_g1+1)
        sin_g2 = (self.y_goal2 - self.y_agent)/(dist_g2+1)
        cos_g2 = (self.x_goal2 - self.x_agent)/(dist_g2+1)
        sin_sw = (self.y_switch - self.y_agent)/(dist_s+1)
        cos_sw = (self.x_switch - self.x_agent)/(dist_s+1)
        return sin_g1, cos_g1, sin_g2, cos_g2, sin_sw, cos_sw
#########################################################################################################
# %%%%%"今の状態が報酬を与える状態か、罰を与える状態かを判定する関数"%%%%%
    def state_check(self) -> Tuple[float, str]:
        dist_g1, dist_g2, dist_s = self.objects_distance()
        if self.x_agent <= 0 or self.field_x <= self.x_agent:# X軸方向の壁に激突した場合
            # print(f'agent: {self.x_agent, self.y_agent}')
            # print(f'goal: {self.x_goal1, self.y_goal1}')
            return self.penalty_wall, "wall"
        elif self.y_agent <= 0 or self.field_y <= self.y_agent:# Y軸方向の壁に激突した場合
            # print(f'agent: {self.x_agent, self.y_agent}')
            # print(f'goal: {self.x_goal1, self.y_goal1}')
            return self.penalty_wall, "wall"
        elif dist_s <= self.radius_switch:# スイッチを踏んだ場合
            self.switch_push_flag = 1
            return 0.0, 'field'
        elif dist_g1 <= self.radius_goal:
            # print(f'agent: {self.x_agent, self.y_agent}')
            # print(f'goal: {self.x_goal1, self.y_goal1}')
            # print(dist_g1)
            if self.switch_push_flag == 1:

                return self.reward_goal, "goal"
            else:
                return self.spena, "out"
        # elif dist_g2 <= self.radius_goal
        #     if self.goal_rl_flag == 1:
        #         return self.reward_goal, "goal"
        #     else:
        #         return -0.5, "out"
        else:
            return 0.0, "field"

    def wall_distance_input(self):
        right_wall = self.field_x - self.x_agent
        left_wall = self.x_agent
        up_wall = self.field_y - self.y_agent
        down_wall = self.y_agent
        return 1/(up_wall+1), 1/(down_wall+1), 1/(right_wall+1), 1/(left_wall+1)

# %%%%%"エージェントが壁に衝突した際の挙動を与える関数"%%%%%
    def agent_crash_wall(self):
        if self.x_agent < 0:
            self.x_agent = 0.0
        if self.x_agent > self.field_x:
            self.x_agent = self.field_x
        if self.y_agent < 0:
            self.y_agent = 0.0
        if self.y_agent > self.field_y:
            self.y_agent = self.field_y

# %%%%%"エージェントの初期位置を変数に代入し、最初の軌跡がリストに格納される関数"%%%%%
    def init_state(self):
        self.x_agent = self.init_x_agent
        self.y_agent = self.init_y_agent
        self.add_p_log(self.x_agent, self.y_agent)
        self.switch_push_flag = 0
# %%%%%"エージェントのみの初期座標を座標格納リストに入れる関数%%%%%
    def init_agent_random(self):
        while True:
            self.x_agent = random.uniform(0.0, self.field_x)
            self.y_agent = random.uniform(0.0, self.field_y)
            dist_g1, dist_g2, dist_s = self.objects_distance()
            if dist_g1 >= self.radius_goal and dist_g2 >= self.radius_goal and dist_s >= self.radius_switch:
                break
        # self.add_p_log(self.x_agent, self.y_agent)
        self.switch_push_flag = 0

    def dist_comp(self, x1, y1, x2, y2):
        return pow(pow(x2-x1, 2.0)+pow(y2-y1, 2.0) ,0.5)
# %%%%%"ゴールとエージェントの初期座標をランダムで決めて座標格納リストに入れる関数"%%%%%
    def all_p_random_set(self):
        # self.x_goal = random.uniform(0.0, self.field_x)
        # self.y_goal = random.uniform(0.0, self.field_y)
        # while True:
        #     self.x_agent = random.uniform(0.0, self.field_x)
        #     self.y_agent = random.uniform(0.0, self.field_y)
        #     if self.goal_distance() >= self.radius_goal:
        #         break
        #
        # self.add_p_log(self.x_agent, self.y_agent)
        self.switch_push_flag = 0
        self.x_goal1 = random.uniform(self.radius_goal, self.field_x - self.radius_goal)
        self.y_goal1 = random.uniform(self.radius_goal, self.field_y - self.radius_goal)
        # while True:
        #     self.x_goal2 = random.uniform(self.radius_goal, self.field_x - self.radius_goal)
        #     self.y_goal2 = random.uniform(self.radius_goal, self.field_y - self.radius_goal)
        #     if self.dist_comp(self.x_goal1, self.y_goal1, self.x_goal2, self.y_goal2) >= self.radius_goal * 2:
        #         break
        while True:
            self.x_switch = random.uniform(self.radius_goal, self.field_x - self.radius_goal)
            self.y_switch = random.uniform(self.radius_goal, self.field_y - self.radius_goal)
            if self.dist_comp(self.x_goal1, self.y_goal1, self.x_switch, self.y_switch) >= self.radius_goal + self.radius_switch:
                    # and self.dist_comp(self.x_goal2, self.y_goal2, self.x_switch,
                    #                    self.y_switch) >= self.radius_goal + self.radius_switch:
                break
        while True:
            self.x_agent = random.uniform(0.0, self.field_x)
            self.y_agent = random.uniform(0.0, self.field_y)
            dist_g1, dist_g2, dist_s = self.objects_distance()
            # todo self.radius_goal+4をself.radius_goalにした
            if dist_g1 >= self.radius_goal \
                    and dist_g2 >= self.radius_goal \
                    and dist_s >= self.radius_switch:
                break
        # self.add_p_log(self.x_agent, self.y_agent)
        # self.swich_push_flag = 0
        # self.replot_goal()
#%%%%%"# 値を制限する関数"%%%%%
    def limit_func(self, val, min_val, max_val):
        ret = (val - min_val) / (max_val - min_val)
        if ret < 0.0:
            return 0.0
        elif ret > 1.0:
            return 1.0
        return ret
    def limit_func2(self, val, min_val, max_val):  # -1.0〜1.0
        ret = ((val - min_val) / (max_val - min_val)) * 2.0 - 1.0
        if ret < -1.0: # この場合は計算上あり得ない
            return -1.0
        elif ret > 1.0:
            return 1.0
        return ret
# %%%%%"現在のエージェントの状態を返す関数(返り値はそのままネットワークの入力情報として使用する)"%%%%%
    def get_state(self):
        # todo 正規化を消して、乱数を入力に加えている。
        # rand = np.random.uniform(-0.1, 0.1, (6,))
        # u_wall, d_wall, r_wall, l_wall = self.wall_distance_input()
        ret_sw1, ret_sw2 = self.switch_check()# switch_check:ゴールを踏まずにちゃんと最初にスイッチに来たか着てないかを判別する関数
        ret_gd1, ret_gd2, ret_sd = self.objects_distance()# gd:goal_distance /sd:switch_distance
        # ret_gd2 = self.limit_func2(ret_gd2, 0.0, pow(self.field_x ** 2 + self.field_y ** 2, 0.5))# 正規化している(-1.0 ~ 1.0)
        # ret_gd1 = self.limit_func2(ret_gd1, 0.0, pow(self.field_x ** 2 + self.field_y ** 2, 0.5))# 正規化している(-1.0 ~ 1.0)
        # ret_sd = self.limit_func2(ret_sd, 0.0, pow(self.field_x ** 2 + self.field_y ** 2, 0.5))# 正規化している(-1.0 ~ 1.0)
        ret_gd1 = 1 / (ret_gd1 + 1)
        ret_sd = 1/(ret_sd+1)
        ret_gsin1, ret_gcos1, ret_gsin2, ret_gcos2, ret_swsin, ret_swcos = self.objects_angle()
        # return ret_sw1, ret_gd1+rand[0], ret_sd+rand[1], ret_gsin1+rand[2], ret_gcos1+rand[3], \
        #        ret_swsin+rand[4], ret_swcos+rand[5]
        return ret_sw1, ret_gd1, ret_sd, ret_gsin1, ret_gcos1, ret_swsin, ret_swcos
        # return ret_sw1, ret_gd1, ret_sd, ret_gsin1, ret_gcos1, ret_swsin, ret_swcos, u_wall, d_wall, r_wall, l_wall
# %%%%%"リアプノフ指数の計測のため、ある試行ごとにagent,goal,switchの位置を固定する関数"%%%%%
#     def fixation_object(self, gp: tuple=(), sp: tuple=(), stock_apx: Optional[Iterable]=None, stock_apy: Optional[Iterable]=None):
#         if [stock_apx, stock_apy] == [None, None]:
#             self.fig = plt.figure(figsize=(6,6))
#             self.ax = self.fig.add_subplot(111)
#             axis([-1, self.field_x + 1, -1, self.field_y + 1])
#             xlabel("x_axis")  # x軸ラベル
#             ylabel("y_axis")  # y軸ラベル
#             title("test")  # グラフのタイトル
#             legend()  # グラフの凡例
#             grid(True)  # グラフのグリッド線の表示
#             self.x_goal1, self.y_goal1 = gp
#             self.x_switch, self.y_switch = sp
#             self.cir_goal1 = plt.Circle(gp, self.radius_goal, ec="#000077", fill=False, lw=4)
#             self.cir_switch = plt.Circle(sp, self.radius_switch, ec="#770000", fill=False, lw=4)
#             self.ax.add_patch(self.cir_goal1)
#             self.ax.add_patch(self.cir_switch)
#         else:
#             self.ax.plot(stock_apx, stock_apy, lw=2, marker=".", alpha=0.7)

    def fixation_objects(self, fx: float, fy: float, ap: Tuple[float, float], gp: Tuple[float, float], sp: Tuple[float, float]):
        self.x_agent, self.y_agent = ap
        self.x_goal1, self.y_goal1 = gp  # ゴールの位置X座標
        self.x_switch, self.y_switch = sp







    
    
        




   





