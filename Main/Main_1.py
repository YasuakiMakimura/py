# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from random import seed
import chainer.functions as F
from chainer import Variable
from chainer import optimizers, serializers
import Directory_Processor
import Field_goal
import Past_Reservoir
import Layered_NN
import Bokeh_line


class Database:
    def __init__(self):
        self.para = {'learning_type': "conventional", 'seed': 1, 'learning_rate': 0.01,
                     'episode': 100, 'limit_step': 200, 'plot_cycle': 1, 'model_savecycle': 1000,
                     'n_neuron': {'in': 7, 'fb': 10}, 'connect_rate': 10,
                     'wscale': {'in': 0.0, 'fb': 1.0},
                     'n_neuron_of_readout': {'l1': 100, 'l2': 40, 'l3': 10, 'top': 3},
                     'lambda': 1.5, 'reward': 0.8, 'penalty_for_wallcrash': -0.1,
                     'penalty_for_non-switching': -0.8, 'switch_flag': 10, 'discount_rate': 0.9,
                     'smoothing_factor': 0.1, 'load_model': False,
                     'load_file': '/home/maki/MEGA/tex/reservoir/RL/2018/6_22/Figure/'
                                  '0622_0757_02/mlr_model/ep49999.npz'}

        self.data = Directory_Processor.DataDirectory()
        with open(f'{self.data.data_path_dict["root"]}para', 'w') as f:
            for key, value in self.para.items():
                f.write(f'{key} = {str(value)}"\n"')


class ModuleSource:
    def __init__(self):
        self.obj = {}

    def __setattr__(self, key, value):
        self.obj[key] = value

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def member(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


class MainSource:
    def __init__(self, seed):
        self.database = Database()
        self.module = ModuleSource
        seed(seed)
        np.random.seed(seed)

    def init_for_episode(self, episode):
        print(f'epi: {episode}')


def main():
    db = Database()
    mod = ModuleSource()
    mod.task = Field_goal.Field(Num_plot=1, fig_number=0, rew=db.para['reward'],
                                pena=db.para['penalty_for_wallcrash'],
                                spena=db.para['penalty_for_non-switching'],
                                sw_scale=db.para['switch_flag'])

    mod.reservoir['1'] = Past_Reservoir.Reservoir(db.para['n_neuron']['in'], 300,
                                                  db.para['n_neuron']['fb'], v_tau=(2,),
                                                  n_tau=(300,), p_connect=db.para['connect_rate'],
                                                  v_lambda=db.para['lambda'],
                                                  in_wscale=db.para['wscale']['in'],
                                                  fb_wscale=db.para['wscale']['fb'])

    mod.reservoir['2'] = Past_Reservoir.Reservoir(db.para['n_neuron']['in'], 700,
                                                  db.para['n_neuron']['fb'], v_tau=(10,),
                                                  n_tau=(700,), p_connect=db.para['connect_rate'],
                                                  v_lambda=db.para['lambda'],
                                                  in_wscale=db.para['wscale']['in'],
                                                  fb_wscale=db.para['wscale']['fb'])

    mod.mlr = Layered_NN.MyChain(1000, db.para['n_neuron']['in'],
                                 db.para['n_neuron_of_readout']['l1'],
                                 db.para['n_neuron_of_readout']['l2'],
                                 db.para['n_neuron_of_readout']['l3'],
                                 db.para['n_neuron_of_readout']['top'])

    optimizer = optimizers.SGD(lr=db.para['learning_rate'])
    optimizer.use_cleargrads()
    optimizer.setup(mod.mlr)
    if db.para['load_model']:
        serializers.load_npz(db.para['load_file'], mod.mlr)
        input(f'model ({db.para["load_file"]}) load, ok? Please Enter')

    ma_x, ma_y = 0, 0
    for epi in range(db.para['episode']):
        print()


if __name__ == "__main__":
    main()
