# coding=utf-8

import numpy as xp
import os
import matplotlib.pyplot as plt
from pylab import *
import random
import itertools as it
from copy import *
import functools
from collections import namedtuple, defaultdict
import pandas as pd
import pickle
from typing import Tuple, List, Iterable, Union, Optional, ClassVar

import chainer
# import cupy as xp
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


# class Test:
#     def __init__(self, database: ClassVar, tsk: ClassVar, net: ClassVar, out: ClassVar):
#         self.db: ClassVar = database
#         self.tsk: ClassVar = tsk
#         self.net: ClassVar = net
#         self.out: ClassVar = out
#         self.i = {}
#         self.out_o = {}
#
#     def forward(self):
#         self.i['sw_in'], self.i['d_g'], self.i['d_sw'], self.i['sin_g'], self.i['cos_g'], self.i['sin_sw'], self.i['cos_sw'] = self.tsk.get_state()
#         in_inf: Tuple[float, ...] = tuple(self.i.values())
#         bypass: Variable[np.ndarray] = Variable(xp.array([in_inf], dtype=np.float32))
#         self.net.in_o[:] = in_inf  # type: np.ndarray
#         self.net.ru_comp()
#         self.net.ro_comp()
#         out_in: Variable[np.ndarray] = Variable(xp.array([self.net.ro], dtype=np.float32))
#         self.out_o['1st'], self.out_o['2nd'], self.out_o['3rd'], self.out_o['top'] = self.out.ff_comp(net_in=out_in, bypass=bypass, net_type=self.db.p['net_type'])
#         self.net.read_o = self.out_o['3rd'].data[0]
#
#     def move(self):
#         self.tsk.move_agent(self.out_o['top'].data[0][1], self.out_o['top'].data[0][2])
#         self.tsk.agent_crash_wall()
#         reward, self.str_state = self.tsk.state_check()  # type: float, str


# noinspection PyTypeChecker
def tes():
    import Main_v0 as main
    # import Main_non_stau as main
    import Field_goal as fd
    import Neu_Reservoir_b2 as neu
    import Net_chainer_ReNN_b4 as mlr
    import Logger as log
    db = main.Database()
    random.seed(db.p['seed'])
    np.random.seed(db.p['seed'])
    xp.random.seed(db.p['seed'])

    out_o = {}
    i = {}
    gp = (db.p['field_x'] / 2, db.p['field_y'] / 2)
    sp = (db.p['field_x'] / 2, db.p['field_y'] - db.p['field_y'] / 10)
    ap = {}
    ap['left'] = (db.p['field_x'] * 1 / 6, db.p['field_y'] / 10)
    ap['left-center'] = (db.p['field_x'] * 2 / 6, db.p['field_y'] / 10)
    ap['center'] = (db.p['field_x'] * 3 / 6, db.p['field_y'] / 10)
    ap['right-center'] = (db.p['field_x'] * 4 / 6, db.p['field_y'] / 10)
    ap['right'] = (db.p['field_x'] * 5 / 6, db.p['field_y'] / 10)
    init_rnd_scale = 0

    net = neu.Reservoir(db.p['n_in'], db.p['n_h_neuron'], db.p['n_fb'], db.p['v_tau'], db.p['n_tau'], p_connect=db.p['p_connect'], v_lambda=db.p['v_lambda'], in_wscale=db.p['net_in_wscale'], fb_wscale=db.p['net_fb_wscale'])
    out = mlr.MyChain(n_in=db.p['n_h_neuron'], n_bypass=db.p['n_in'], n_l1=db.p['out_l1'], n_l2=db.p['out_l2'], n_l3=db.p['out_l3'], n_top=db.p['out_top'], nor_w=db.p['out_lecnor_ws'], uni_w=db.p['out_uni_ws'], uni_bypass_w=db.p['bypass_ws'], nobias=db.p['mlr_nobias'])
    optimizer = optimizers.SGD(lr=db.p["learning_rate"])
    # noinspection PyDeprecation
    optimizer.use_cleargrads()
    optimizer.setup(out)
    if db.p['load_w'][0]:
        serializers.load_npz(db.p['load_w'][1], out)
    tsk = fd.Field(field_x=db.p['field_x'], field_y=db.p['field_y'], Num_plot=len(ap.keys()), fig_number=0, rew=db.p['reward'], pena=db.p['wall_penalty'], spena=db.p['pena_without_switching'], testing=True, sw_scale=db.p['sw_scale'])
    # lc = log.Log(n_line=2, num_fig=1, c=('red', 'blue'), xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep'], auto_xrange=False)
    # lc = log.Log(n_line=1, num_fig=1, c='blue', xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep'], auto_xrange=False)
    # test_lc = log.Log(n_line=2, num_fig=2, c=('red', 'blue'), xlabel='episode', ylabel='step', y_high=db.p['max_step'] + 25, x_high=db.p['n_ep_test'], auto_xrange=False)
    critic = log.Log(n_line=2, num_fig=3, y_low=-1.1, y_high=1.1, x_high=20, xlabel='step',
                     ylabel='critic', c=('magenta', 'black'), origin_line=True, any_line=True,
                     iter_any=[db.p['reward']] * 20, auto_xrange=True, grid=True)
    stau = log.Log(n_line=db.p['n_pl_stau'], num_fig=4, subplot=True, y_low=-1.1, y_high=1.1,
                   figsize=(8, 6), x_high=20, inf=db.p['v_tau'][1], c='red', xlabel='episode',
                   ylabel='output(τ: 2)')
    ltau = log.Log(n_line=db.p['n_pl_ltau'], num_fig=5, subplot=True, y_low=-1.1, y_high=1.1,
                   figsize=(8, 6), x_high=20, inf=db.p['v_tau'][0], xlabel='episode',
                   ylabel='output(τ: 10)')
    fb = log.Log(n_line=db.p['out_l3'], num_fig=6, subplot=True, y_low=-1.1, y_high=1.1, figsize=(8, 6), x_high=db.p['max_step'], c='green', xlabel='episode', ylabel='output(feedback)')
    d_sw = log.Log(n_line=1, num_fig=7, y_low=-1.1, y_high=1.1, x_high=200, c='#FA8072', xlabel='step', ylabel='input of ds', grid=True)
    sw_in = log.Log(n_line=1, num_fig=8, y_low=-db.p['sw_scale']/10, y_high=db.p[
                                                                                'sw_scale']+db.p[
        'sw_scale']/10, x_high=20, c='#EE82EE', xlabel='step', ylabel='input of switch', grid=True)
    actor = log.Log(n_line=2, num_fig=9, y_low=-0.9, y_high=0.9, x_high=20, c=('#FF6347',
                                                                               '#008080'), xlabel='step', ylabel='actor', legend=('Actor x', 'Actor y'), grid=True)
    td = log.Log(n_line=2, num_fig=7, y_low=0, y_high=1.1, x_high=db.p['n_ep'], c=('#000080', '#FF4500'), xlabel='episode', ylabel='exploration component', auto_xrange=False, legend=('Actor x', 'Actor y'), alpha=(0.5, 0.5), grid=True)
    exp = log.Log(n_line=2, num_fig=11, y_low=0, y_high=1.1, x_high=db.p['n_ep'], c=('#000080', '#FF4500'), xlabel='episode', ylabel='exploration component', auto_xrange=False, legend=('Actor x', 'Actor y'), alpha=(0.5, 0.5), grid=True)
    # n_step = log.Log(n_line=2, num_fig=9, y_low=-0.9, y_high=0.9, x_high=200, c=('#FF6347', '#008080'), xlabel='step', ylabel='actor', legend=('Actor x', 'Actor y'), grid=True)
    # w_actor = log.Log(n_line=10, num_fig=10, y_low=-5.0, y_high=5.0, x_high=db.p['max_step'], c=[None for i in 10], xlabel='step', ylabel='weight of actor x')
    # tes = Test(database=db, tsk=tsk, net=net, out=out)

    # gp: Tuple[float, float] = (float(input('goal x ?: ')), float(input('goal y ?: ')))te
    # sp: Tuple[float, float] = (float(input('switch x ?: ')), float(input('switch y ?: ')))

    for num_ep, ep in enumerate(ap.keys()):
        # ap: Tuple[float, float] = (float(input('agent x ?: ')), float(input('agent y ?: ')))
        print(f'ep: {ep}')
        net.reset_net()
        tsk.fixation_objects(fx=db.p['field_x'], fy=db.p['field_y'], ap=ap[ep], gp=gp, sp=sp)
        tsk.ini_add_p_log_test(posi_x=tsk.x_agent, posi_y=tsk.y_agent, var_ep=num_ep)
        list_stau = [[] for j in range(db.p['n_pl_stau'])]
        list_ltau = [[] for j in range(db.p['n_pl_ltau'])]
        list_fb = [[] for j in range(db.p['out_l3'])]
        list_critic = []
        list_ideal_critic = []
        list_d_sw = []
        list_sw = []
        list_actor = [[], []]
        for step in range(db.p['max_step']):
            i['sw_in'], i['d_g'], i['d_sw'], i['sin_g'], i['cos_g'], i['sin_sw'], i['cos_sw'] = tsk.get_state()
            list_d_sw.append(i['d_sw'])
            list_sw.append(i['sw_in'])

            in_inf: Tuple[float, ...] = tuple(i.values())
            bypass: Variable[np.ndarray] = Variable(xp.array([in_inf], dtype=np.float32))
            net.in_o[:] = in_inf  # type: np.ndarray
            net.ru_comp()
            net.ro_comp()
            out_in: Variable[np.ndarray] = Variable(xp.array([net.ro], dtype=np.float32))
            out_o['1st'], out_o['2nd'], out_o['3rd'], out_o['top'] = out.ff_comp(net_in=out_in, bypass=bypass, net_type=db.p['net_type'])
            net.read_o = out_o['3rd'].data[0]
            if db.p['moving_ave']:
                if step == 0:
                    mma_x, mma_y = db.p['alpha'] * out_o['top'].data[0][1], db.p['alpha'] * out_o['top'].data[0][2]
                else:
                    old_mma_x, old_mma_y = mma_x, mma_y
                    mma_x = (1 - db.p['alpha']) * old_mma_x + db.p['alpha'] * out_o['top'].data[0][1]
                    mma_y = (1 - db.p['alpha']) * old_mma_y + db.p['alpha'] * out_o['top'].data[0][2]
                rndx, rndy = out_o['top'].data[0][1] - mma_x, out_o['top'].data[0][2] - mma_y
                if db.p['add_exploration']:
                    tsk.move_agent(out_o['top'].data[0][1] + rndx, out_o['top'].data[0][2] + rndy)
                else:
                    tsk.move_agent(out_o['top'].data[0][1], out_o['top'].data[0][2])
            else:
                # rndx, rndy = random.uniform(-init_rnd_scale, init_rnd_scale), random.uniform(-init_rnd_scale, init_rnd_scale)
                rndx, rndy = np.random.uniform(-init_rnd_scale, init_rnd_scale, (2,))
                tsk.move_agent(out_o['top'].data[0][1] + rndx, out_o['top'].data[0][2] + rndy)
            list_actor[0].append(out_o['top'].data[0][1])
            list_actor[1].append(out_o['top'].data[0][2])
            tsk.agent_crash_wall()
            reward, str_state = tsk.state_check()  # type: float, str
            tsk.add_p_log2(posi_x=tsk.x_agent, posi_y=tsk.y_agent)
            for sss in range(db.p['n_pl_stau']):
                list_stau[sss].append(net.ro[sss])
            for lll in range(-1, -db.p['n_pl_ltau'] - 1, -1):
                list_ltau[lll].append(net.ro[lll-10])
            for fff in range(db.p['out_l3']):
                list_fb[fff].append(out_o['3rd'].data[0][fff])
            list_critic.append(out_o['top'].data[0][0])
            if str_state in {'goal', 'out'}:
                break
        """STEP_END"""
        # tsk.replot2()
        list_ideal_critic = [db.p['reward'] * (db.p['discount_rate'] ** i) for i in range(step + 1)]
        list_ideal_critic.sort()
        tsk.replot_test(var_ep=num_ep)
        tsk.replot_goal()

        stau.subplot(list_stau)
        ltau.subplot(list_ltau)
        fb.subplot(list_fb)
        stau.save(nm_fig=db.p[f'path_dir_tau{stau.k["inf"]}'] + f'{ep}_tau{stau.k["inf"]}')
        ltau.save(nm_fig=db.p[f'path_dir_tau{ltau.k["inf"]}'] + f'{ep}_tau{ltau.k["inf"]}')
        fb.save(nm_fig=f'{db.p["path_dir_fb"]}fb_{ep}')
        # critic.plot(list_data=list_critic)
        critic.multi_plot(lists_data=(list_critic, list_ideal_critic))
        critic.save(nm_fig=f'{db.p["path_dir_critic"]}critic_{ep}')
        # d_sw.plot(list_data=list_d_sw)
        # d_sw.save(nm_fig=f'{db.p["path_dir_data"]}ds_{ep}')
        sw_in.plot(list_data=list_sw)
        sw_in.save(nm_fig=f'{db.p["path_dir_sw_in"]}sw_in_{ep}')
        actor.multi_plot(lists_data=(list_actor[0], list_actor[1]))
        actor.save(nm_fig=f'{db.p["path_dir_actor"]}actor_{ep}')
    """EPISODE_END"""
    tsk.save(fig_name=f'{db.p["path_dir_traj"]}testing_traj')


if __name__ == "__main__":
    tes()
