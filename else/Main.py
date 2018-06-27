# coding=utf-8

import os
import zipfile
from time import perf_counter
from collections import defaultdict
from typing import Tuple, List, Union, ClassVar, Any
from pprint import pprint
from numba import jit

import numpy as xp
import chainer
# import cupy as xp
from pylab import *

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


import Field_goal as fd
import Neu_Reservoir as neu
import Layered_NN as mlr
import Logger as log


def min_max(val: object, min_val: object, max_val: object) -> object:
    if val < min_val:  # ある値が下限以下である場合
        return min_val
    elif val > max_val:  # ある値が上限以上である場合
        return max_val
    return val  # ある値がちゃんと指定した値域内に収まる場合はそのままある値を返す


def zip_getfolder(zip_name: str, walk_path: str, mode: str = 'w'):
    with zipfile.ZipFile(f'{zip_name}.zip', mode, compression=zipfile.ZIP_DEFLATED) as myzip:
        for dirpath, dirname, filename in os.walk(walk_path):  # type: str, List[str, ...], List[str, ...]
            # myzip.write(dirpath)
            # for file in filename:
            print(list(os.walk(walk_path)))
            # for len_dirname in dirname:
            for len_filename in filename[1]:
                # print(len_dirname)
                # print(type(len_dirname))
                myzip.write(os.path.join(dirname[0], len_filename))


# noinspection PyMissingOrEmptyDocstring
# class Storage:
#     def __init__(self, *activity: ClassVar, task: ClassVar, lc: ClassVar, cycle_plt: int):
#         self.act: Tuple[ClassVar] = activity
#         self.t: ClassVar = task
#         self.lc: ClassVar = lc
#         self.cycle_plt = cycle_plt
#         self.list_finish_step = []
#
#     def ini_p_log(self, var_ep):
#         if var_ep % self.cycle_plt == 0:
#             self.t.ini_add_p_log(posi_x=self.t.x_agent, posi_y=self.t.y_agent)
#
#     def p_log(self, var_ep):
#         if var_ep % self.cycle_plt == 0:
#             self.t.add_p_log2(posi_x=self.t.x_agent, posi_y=self.t.y_agent)
#
#     def step_log(self, num_finish_step):
#         self.list_finish_step.append(num_finish_step)


# class Save:
#     def __init__(self, **kwargs):
#         self.nm_fig = {}
#         self.nm_fig['nm_traj_file']: str = 'traj'
#         # cls.nm_fig['active'] = ''
#
#         for key_kwg, v_kwg in kwargs.items():
#             try:
#                 self.nm_fig[key_kwg]
#             except KeyError:
#                 print(f'"""" List of **kwargs.keys() """": {self.nm_fig.keys()}\n')
#                 raise AttributeError(f'arg("{key_kwg}") does not exist in kwargs of class ("{self.__class__.__name__}")')
#             else:
#                 self.nm_fig[key_kwg] = v_kwg
#         # return super().__new__(cls)


# noinspection PyMissingOrEmptyDocstring
# class Plot(Storage, Save):
#     # def __new__(cls, *activity: ClassVar, task: ClassVar, lc: ClassVar, cycle_plt: int, **nm_fig):
#     #     cls.nm_fig = defaultdict(lambda: 'none')
#     #     cls.nm_fig['nm_traj_file']: str = 'traj'
#     #     # cls.nm_fig['active'] = ''
#     #
#     #     for key_kwg, v_kwg in nm_fig.items():
#     #         if cls.nm_fig[key_kwg] == 'none':
#     #             print(f'**nm_fig.keys(): {nm_fig.keys()}')
#     #             raise AttributeError(f'not exist arg("{key_kwg}") in kwargs of class ("{cls.__name__}")')
#     #         cls.nm_fig[key_kwg] = v_kwg
#     #     return super().__new__(cls)
#
#     def __init__(self, *activity: ClassVar, task: ClassVar, lc: ClassVar, cycle_plt: int, **kwargs):  # todo タスク関係,　Critic, 出力関係をkwargにインスタンスする
#         print(kwargs)
#         super().__init__(activity, task=task, lc=lc, cycle_plt=cycle_plt)
#         super().__init__(kwargs=kwargs)
#
#     def tsk(self, var_ep):
#         # print(self.cycle_plt)
#         # print(f'%%: {var_ep % self.cycle_plt}')
#         if var_ep % self.cycle_plt == 0:
#             self.t.replot2()
#             self.t.replot_goal()
#             self.t.save(fig_name=f'{self.nm_fig["nm_traj_file"]}')
#
#     def activity(self, *list_activity: List[float]):
#         if len(self.act) == len(list_activity):
#             raise IndexError(f'len(self.act): {len(self.act)} != len(list_activity): {len(list_activity)}')
#         for len_act in range(len(self.act)):
#             self.act[len_act].update_all(list_activity[len_act])  # todo 入力情報　getter
#             self.act[len_act].plot()
#
#     def critic(self):
#         pass
#
#     def lcurve(self):  # todo learning curve
#         # self.lc.log_step_list2(self.list_finish_step)
#         # self.lc.plot(save_im=True, y_low=0, y_high)
#         self.lc.plot(self.list_finish_step)
#
#     def mma(self):  # todo 出力と移動平均のグラフ
#         pass


# noinspection PyMissingOrEmptyDocstring,PyUnresolvedReferences,PyAttributeOutsideInit,PyShadowingNames
class Train:
    def __init__(self, database: ClassVar, tsk: ClassVar, net: ClassVar, out: ClassVar, external_rnd: bool):
        self.db: ClassVar = database
        self.tsk: ClassVar = tsk
        self.net: ClassVar = net
        self.out: ClassVar = out
        self.out_o = {}
        self.i = {}
        self.pre_i = {}
        if external_rnd:
            self.init_rnd_scale = self.db.p['init_rnd_scale']
            self.gene_rnd = self.gene_rnd_ext
        else:
            self.gene_rnd = self.gene_rnd_mma

    # @jit(nopython=True)
    def init_step(self):
        # input('ini_setting')
        self.net.reset_net()
        self.tsk.all_p_random_set()
        self.i['sw_in'], self.i['d_g'], self.i['d_sw'], self.i['sin_g'], self.i['cos_g'], self.i['sin_sw'], self.i['cos_sw'] = self.tsk.get_state()
        self.in_inf: Tuple[float, ...] = tuple(self.i.values())
        self.bypass: Variable[np.ndarray] = Variable(xp.array([self.in_inf], dtype=np.float32))
        self.net.in_o[:] = self.in_inf  # type: np.ndarray
        self.net.ru_comp()
        self.net.ro_comp()
        self.out_in: Variable[np.ndarray] = Variable(xp.array([self.net.ro], dtype=np.float32))



        # print(self.tsk.x_agent, self.tsk.y_agent)
        # print(self.tsk.x_goal1, self.tsk.y_goal1)
        # print(self.tsk.x_switch, self.tsk.y_switch)
        # print('\n')

    # @jit(nopython=True)
    def forward(self):
        # pprint(f'in_inf: {self.in_inf}')
        # pprint(f'bypass: {self.bypass}')
        # pprint(f'net.in_o: {self.net.in_o}')
        # pprint(f'net.ru: {self.net.ru}')
        # pprint(f'net.ro: {self.net.ro}')
        # pprint(f'out_in(vari): {self.out_in}')
        self.out_o['1st'], self.out_o['2nd'], self.out_o['3rd'], self.out_o['top'] \
            = self.out.ff_comp(net_in=self.out_in, bypass=self.bypass, net_type=self.db.p['net_type'])
        # pprint(f'out_o: {self.out_o}')
        self.net.read_o = self.out_o['3rd'].data[0]
        # pprint(f'net.read_o: {self.net.read_o}')

    # @jit(nopython=True)
    # noinspection PyUnusedLocal
    def gene_rnd_ext(self):
        self.rndx, self.rndy = xp.random.uniform(-self.init_rnd_scale, self.init_rnd_scale, (2,))
        # pprint(f'rndx: {self.rndx}, rndy: {self.rndy}')

    # @jit(nopython=True)
    def gene_rnd_mma(self, var_step: int):
        if var_step == 0:
            self.mma_x, self.mma_y = self.out_o['top'].data[0][1], self.out_o['top'].data[0][2]
        else:
            self.old_mma_x, self.old_mma_y = self.mma_x, self.mma_y
            self.mma_x = self.db.p['alpha'] * self.out_o['top'].data[0][1]
            self.mma_y = self.db.p['alpha'] * self.out_o['top'].data[0][2]
            self.mma_x += (1 - self.db.p['alpha']) * self.old_mma_x
            self.mma_y += (1 - self.db.p['alpha']) * self.old_mma_y
        self.rndx = self.out_o['top'].data[0][1] - self.mma_x
        self.rndy = self.out_o['top'].data[0][2] - self.mma_y

    # @jit(nopython=True)
    def move(self):
        self.tsk.move_agent(self.out_o['top'].data[0][1] + self.rndx, self.out_o['top'].data[0][2] + self.rndy)
        # print(f'movex: {self.out_o["top"].data[0][1] + self.rndx}')
        # print(f'movey: {self.out_o["top"].data[0][2] + self.rndy}')
        self.tsk.agent_crash_wall()
        self.reward, self.str_state = self.tsk.state_check()  # type: float, str


    def prediction(self):
        # print(f'\n""""""""PRE""""""""""\n')
        self.i['sw_in'], self.i['d_g'], self.i['d_sw'], self.i['sin_g'], self.i['cos_g'], self.i['sin_sw'], self.i['cos_sw'] = self.tsk.get_state()
        # print(f'rew: {self.reward}, str_state: {self.str_state}')
        # self.reward, self.str_state = self.tsk.state_check()  # type: Tuple[float, str]
        # print(f'rew: {self.reward}, str_state: {self.str_state}')
        self.in_inf: Tuple[float, ...] = tuple(self.i.values())

        # print(f'pre_in_inf: {self.pre_in_inf}')
        self.bypass: Variable[np.ndarray] = Variable(xp.array([self.in_inf], dtype=np.float32))
        self.net.in_o[:] = self.in_inf  # type: np.ndarray
        # print(f'pre_bypass: {self.pre_bypass}')
        # print(f'net.ro(before): {self.net.ro}')
        # print(f'net.read(before): {self.net.read_o}')
        # self.net.pre_ru_comp(in_o=xp.array(self.in_inf))
        self.net.ru_comp()
        # print(f'pre_ru: {self.net.pre_ru}')
        # print(f'net.ru(2): {self.net.ru}')
        self.net.ro_comp()
        # print(f'pre_ro: {self.net.pre_ro}')
        # print(f'net.ro(before): {self.net.ro}')
        # print(f'net.read(before): {self.net.read_o}')
        self.out_in: Variable[np.ndarray] = Variable(xp.array([self.net.ro], dtype=np.float32))
        # print(f'pre_out_in: {self.pre_out_in}')
        self.out_o['pre_1st'], self.out_o['pre_2nd'], self.out_o['pre_3rd'], self.out_o['pre_top'] \
            = self.out.ff_comp(net_in=self.out_in, bypass=self.bypass, net_type=self.db.p['net_type'])
        # print(f'out_o(2): {self.out_o}')

    def update(self):
        if self.str_state in {'goal', 'out'}:
            self.out_o['pre_top'] = Variable(xp.array([[0.0]], dtype=np.float32))
        self.td = self.reward + self.db.p['discount_rate'] * self.out_o['pre_top'].data[0][0] - self.out_o['top'].data[0][0]
        print(f'td: {self.td}')
        self.t_critic = self.out_o['top'].data[0][0] + self.td
        self.t_actorx = self.out_o['top'].data[0][1] + self.rndx * self.td
        self.t_actory = self.out_o['top'].data[0][2] + self.rndy * self.td
        self.teach = Variable(xp.array([[self.t_critic,
                                         min_max(self.t_actorx, -self.db.p['t_actor_limit'], self.db.p['t_actor_limit']),
                                         min_max(self.t_actory, - self.db.p['t_actor_limit'], self.db.p['t_actor_limit'])]],
                                       dtype=np.float32))  # todo ここはxpじゃなくてnpで良いのか
        self.out.cleargrads()
        loss = F.mean_squared_error(self.out_o['top'], self.teach)
        loss.backward()
        optimizer.update()  # todo stepのloop外の処理も関数としてかく？ todo saveの関数を作る？


class Test:
    def __init__(self, database: ClassVar, tsk: ClassVar, net: ClassVar, out: ClassVar):
        self.db: ClassVar = database
        self.tsk: ClassVar = tsk
        self.net: ClassVar = net
        self.out: ClassVar = out
        self.i = {}
        self.out_o = {}

    def forward(self):
        self.i['sw_in'], self.i['d_g'], self.i['d_sw'], self.i['sin_g'], self.i['cos_g'], self.i['sin_sw'], self.i['cos_sw'] = self.tsk.get_state()
        self.in_inf: Tuple[float, ...] = tuple(self.i.values())
        self.bypass: Variable[np.ndarray] = Variable(xp.array([self.in_inf], dtype=np.float32))
        self.net.in_o[:] = self.in_inf  # type: np.ndarray
        self.net.ru_comp()
        self.net.ro_comp()
        self.out_in: Variable[np.ndarray] = Variable(xp.array([self.net.ro], dtype=np.float32))
        self.out_o['1st'], self.out_o['2nd'], self.out_o['3rd'], self.out_o['top'] = self.out.ff_comp(net_in=self.out_in, bypass=self.bypass, net_type=self.db.p['net_type'])
        self.net.read_o = self.out_o['3rd'].data[0]

    def move(self):
        self.tsk.move_agent(self.out_o['top'].data[0][1], self.out_o['top'].data[0][2])
        self.tsk.agent_crash_wall()
        self.reward, self.str_state = self.tsk.state_check()  # type: float, str


# noinspection PyMissingOrEmptyDocstring,PyBroadException
class Database:
    def __init__(self, nm_dir_datalist: str = 'data', b_save: bool = True):
        self.p = defaultdict(lambda: 'none')
        os.makedirs(nm_dir_datalist, exist_ok=True)
        files = os.listdir(f'{nm_dir_datalist}/')
        print(f'List of "folder" in ###{nm_dir_datalist}###/: ')
        for file in files:
            print(file)
        try:
            self.nm_dir_data = input(f"\n Name of data folder: ")
            os.makedirs(f'{nm_dir_datalist}/{self.nm_dir_data}')
        except Exception as error:
            print(f'{type(error)}, Please appoint name of data folder, again')
            self.nm_dir_data = input(f"Name of data folder: ")
            os.makedirs(f'{nm_dir_datalist}/{self.nm_dir_data}')
        finally:
            self.p['path_dir_data'] = f'{nm_dir_datalist}/{self.nm_dir_data}/'
        self.p['path_dir_traj'] = f'{self.p["path_dir_data"]}traj/'
        os.makedirs(self.p['path_dir_traj'], exist_ok=True)
        self.p['path_dir_ro'] = f'{self.p["path_dir_data"]}ro/'
        os.makedirs(self.p['path_dir_ro'], exist_ok=True)
        self.p['path_dir_lc'] = f'{self.p["path_dir_data"]}lc/'
        os.makedirs(self.p['path_dir_lc'], exist_ok=True)

        self.p['pl_cycle'] = 500
        self.p['n_pl_stau'] = 10
        self.p['n_pl_ltau'] = 10
        self.p['n_ep'] = 5000
        self.p['n_ep_test'] = 7
        self.p['max_step'] = 200
        self.p['n_in'] = 7
        self.p['n_h_neuron'] = 1000
        self.p['n_fb'] = 10
        self.p['p_connect'] = 10
        self.p['v_lambda'] = 1.8
        self.p['v_tau']: Tuple[Union[int, float]] = (10, 2)
        for off_v_tau in self.p['v_tau']:
            self.p[f'path_dir_tau{off_v_tau}'] = f'{self.p["path_dir_ro"]}tau{off_v_tau}/'
            os.makedirs(self.p[f'path_dir_tau{off_v_tau}'], exist_ok=True)
        self.p['path_dir_fb'] = f'{self.p["path_dir_ro"]}fb/'
        os.makedirs(self.p['path_dir_fb'], exist_ok=True)

        self.p['n_tau']: Tuple[int] = (500, 500)
        self.p['net_in_wscale'] = 1.0
        self.p['net_fb_wscale'] = 1.0
        self.p['net_type'] = 1

        self.p['out_l1'] = 100
        self.p['out_l2'] = 40
        self.p['out_l3'] = self.p['n_fb']
        self.p['out_top'] = 3

        self.p['field_x'] = 16.0
        self.p['field_y'] = 16.0

        self.p['seed'] = 2
        self.p['init_rnd_scale'] = 1.5
        self.p['finish_rnd_scale'] = 1.0
        self.p['learning_rate'] = 0.9
        self.p['alpha'] = 0.2
        if self.p['alpha'] >= 1.0:
            raise ValueError('Not smoothing_coefficient >= 1.0')
        self.p['reward'] = 0.8
        self.p['wall_penalty'] = -0.1
        self.p['pena_without_switching'] = -1.0
        self.p['discount_rate'] = 0.99
        self.p['t_actor_limit'] = 0.8
        self.p['sw_scale'] = 10.0  # todo 日付保存,csvファイルで保存または画像をパラメータと一緒に画像として保存、メモもできるようにする

        self.p['external_rnd'] = True

        if b_save:
            daytime = datetime.datetime.today()
            self.f = open(f'{self.p["path_dir_data"]}para', "a")
            self.f.write("\n\n")
            # self.f.write("exp number:" + str(self.exp_num))
            # self.f.write("############################\n")
            self.f.write(str(daytime) + "\n")
            for k, v in self.p.items():
                self.f.write(k + "=" + str(v) + "\n")
            self.f.close()


def test(nm_npz: str, net: ClassVar, out: ClassVar):
    global step
    db = Database(nm_dir_datalist='test')  # todo saveを最後でやる？　途中でパラメータを変えれるようにするため？
    np.random.seed(db.p['seed'])
    xp.random.seed(db.p['seed'])

    # net = neu.Reservoir(db.p['n_in'], db.p['n_h_neuron'], db.p['out_l3'], db.p['v_tau'], db.p['n_tau'], p_connect=db.p['p_connect'], v_lambda=db.p['v_lambda'], in_wscale=db.p['net_in_wscale'], fb_wscale=db.p['net_fb_wscale'])
    # out = mlr.MyChain(n_in=db.p['n_h_neuron'], n_bypass=db.p['n_in'], n_l1=db.p['out_l1'], n_l2=db.p['out_l2'], n_l3=db.p['out_l3'], n_top=db.p['out_top'])
    optimizer = optimizers.SGD(lr=db.p["learning_rate"])
    # noinspection PyDeprecation
    optimizer.use_cleargrads()
    optimizer.setup(out)
    serializers.load_npz(nm_npz, out)
    test_tsk = fd.Field(field_x=db.p['field_x'], field_y=db.p['field_y'], Num_plot=1, fig_number=0, rew=db.p['reward'], pena=db.p['wall_penalty'], spena=db.p['pena_without_switching'])
    test_stau = log.Log(n_fig=db.p['n_pl_stau'], num_fig=4, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], inf=db.p['v_tau'][1], c='red')
    test_ltau = log.Log(n_fig=db.p['n_pl_ltau'], num_fig=5, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], inf=db.p['v_tau'][0])
    test_fb = log.Log(n_fig=db.p['out_l3'], num_fig=6, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], c='green')

    test = Test(database=db, tsk=test_tsk, net=net, out=out)
    list_finish_step = []
    gp: Tuple[float, float] = (float(input('goal x ?: ')), float(input('goal y ?: ')))
    sp: Tuple[float, float] = (float(input('switch x ?: ')), float(input('switch y ?: ')))
    for ep in range(db.p['n_ep_test']):
        print(f'ep:{ep}')
        ap: Tuple[float, float] = (float(input('agent x ?: ')), float(input('agent y ?: ')))
        net.reset_net()
        test_tsk.fixation_objects(fx=db.p['field_x'], fy=db.p['field_y'], ap=ap, gp=gp, sp=sp)
        test_tsk.ini_add_p_log(posi_x=test_tsk.x_agent, posi_y=test_tsk.y_agent)
        list_stau = [[] for j in range(db.p['n_pl_stau'])]
        list_ltau = [[] for j in range(db.p['n_pl_ltau'])]
        list_fb = [[] for j in range(db.p['out_l3'])]
        for step in range(db.p['max_step']):
            test.forward()
            test.move()
            test_tsk.add_p_log2(posi_x=test_tsk.x_agent, posi_y=test_tsk.y_agent)
            for sss in range(db.p['n_pl_stau']):
                list_stau[sss].append(net.ro[sss])
            for lll in range(-1, -db.p['n_pl_ltau'] - 1, -1):
                list_ltau[lll].append(net.ro[lll])
            for fff in range(db.p['out_l3']):
                list_fb[fff].append(test.out_o['3rd'].data[0][fff])
            if test.str_state in {'goal', 'out'}:
                break
        test_tsk.replot2()
        test_tsk.replot_goal()
        test_tsk.save(fig_name=f'{db.p["path_dir_traj"]}ep{ep}')
        test_stau.subplot(list_stau)
        test_ltau.subplot(list_ltau)
        test_fb.subplot(list_fb)
        test_stau.save(nm_fig=db.p[f'path_dir_tau{test_stau.k["inf"]}'] + f'ep{ep}')
        test_ltau.save(nm_fig=db.p[f'path_dir_tau{test_ltau.k["inf"]}'] + f'ep{ep}')
        test_fb.save(nm_fig=f'{db.p["path_dir_fb"]}fep{ep}')
        list_finish_step.append(step)


def main():
    global step
    db = Database()  # todo saveを最後でやる？　途中でパラメータを変えれるようにするため？
    np.random.seed(db.p['seed'])
    xp.random.seed(db.p['seed'])


    net = neu.Reservoir(db.p['n_in'], db.p['n_h_neuron'], db.p['out_l3'], db.p['v_tau'], db.p['n_tau'], p_connect=db.p['p_connect'], v_lambda=db.p['v_lambda'], in_wscale=db.p['net_in_wscale'], fb_wscale=db.p['net_fb_wscale'])
    out = mlr.MyChain(n_in=db.p['n_h_neuron'], n_bypass=db.p['n_in'], n_l1=db.p['out_l1'], n_l2=db.p['out_l2'], n_l3=db.p['out_l3'], n_top=db.p['out_top'])
    optimizer = optimizers.SGD(lr=db.p["learning_rate"])
    # noinspection PyDeprecation
    optimizer.use_cleargrads()
    optimizer.setup(out)
    tsk = fd.Field(field_x=db.p['field_x'], field_y=db.p['field_y'], Num_plot=1, fig_number=0, rew=db.p['reward'], pena=db.p['wall_penalty'], spena=db.p['pena_without_switching'])
    lc = log.Log(n_fig=1, num_fig=3, c='red', xlabel='episode', ylabel='step', y_high=db.p['max_step'], x_high=db.p['n_ep'])
    stau = log.Log(n_fig=db.p['n_pl_stau'], num_fig=4, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], inf=db.p['v_tau'][1], c='red')
    ltau = log.Log(n_fig=db.p['n_pl_ltau'], num_fig=5, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], inf=db.p['v_tau'][0])
    fb = log.Log(n_fig=db.p['out_l3'], num_fig=6, y_low=-1.1, y_high=1.1, figsize=(8, 4), x_high=db.p['max_step'], c='green')

    train = Train(database=db, tsk=tsk, net=net, out=out, external_rnd=db.p['external_rnd'])
    list_finish_step = []
    for ep in range(db.p['n_ep']):
        print(f'ep:{ep}')
        train.init_step()
        if ep % db.p['pl_cycle'] == 0:
            tsk.ini_add_p_log(posi_x=tsk.x_agent, posi_y=tsk.y_agent)
            list_stau = [[] for j in range(db.p['n_pl_stau'])]
            list_ltau = [[] for j in range(db.p['n_pl_ltau'])]
            list_fb = [[] for j in range(db.p['out_l3'])]
        for step in range(db.p['max_step']):
            # print(f'############STEP{step}#############')
            # input('before_forward')
            train.forward()
            # input('before_gene_rnd')
            train.gene_rnd()
            # input('before_move')
            train.move()
            if ep % db.p['pl_cycle'] == 0:
                tsk.add_p_log2(posi_x=tsk.x_agent, posi_y=tsk.y_agent)
                for sss in range(db.p['n_pl_stau']):
                    list_stau[sss].append(net.ro[sss])
                for lll in range(-1, -db.p['n_pl_ltau'] - 1, -1):
                    list_ltau[lll].append(net.ro[lll])
                for fff in range(db.p['out_l3']):
                    list_fb[fff].append(train.out_o['3rd'].data[0][fff])
            # input('before_pre')
            train.prediction()
            # input('before_update')
            train.update(optimizer=optimizer)
            # print(train.str_state)
            # print('\n')
            if train.str_state in {'goal', 'out'}:
                if train.str_state == 'out':
                    step = 199
                break  # print('\n\n')
        # input('finish_step')
        """STEP_END"""
        print(f'step: {step}, str_state: {train.str_state}, rnd_scale: {train.init_rnd_scale}\n')
        train.init_rnd_scale = train.init_rnd_scale * pow(db.p['finish_rnd_scale'] / db.p['init_rnd_scale'], 1.0 / db.p['n_ep'])
        # print(tsk.x_goal1, tsk.y_goal1)
        # print(tsk.traj_x)
        # print(tsk.traj_y)
        if ep % db.p['pl_cycle'] == 0:
            tsk.replot2()
            # input('before replot_goal')
            tsk.replot_goal()
            # print(tsk.x_goal1, tsk.y_goal1)
            tsk.save(fig_name=f'{db.p["path_dir_traj"]}ep{ep}')
            stau.subplot(list_stau)
            ltau.subplot(list_ltau)
            fb.subplot(list_fb)
            stau.save(nm_fig=db.p[f'path_dir_tau{stau.k["inf"]}'] + f'ep{ep}')
            ltau.save(nm_fig=db.p[f'path_dir_tau{ltau.k["inf"]}'] + f'ep{ep}')
            fb.save(nm_fig=f'{db.p["path_dir_fb"]}fep{ep}')
        list_finish_step.append(step)  # input('finish_ep')
    lc.plot(list_data=list_finish_step)
    lc.save(nm_fig=f'{db.p["path_dir_lc"]}learning_curve')
    nm_npz = input('npz name ?: ')
    serializers.save_npz(nm_npz, out)
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    test(nm_npz=nm_npz, net=net, out=out)


if __name__ == "__main__":
    main()
