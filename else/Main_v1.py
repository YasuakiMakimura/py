# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from datetime import datetime

import Field_goal as field
import Neu_Reservoir_b2 as res
import Layered_NN as mlr
import Logger as log
import Directory_Processor as director

class Database:
    p = {
         'conventional': True,
         'non-emphasized': False,
         'emphasized': False,
         'seed': 1,
         'switch_flag': 10,
         'episode': 50000,
         'limit_step': 200,
         'plot_cycle': 1000,
         'savecycle_mode1': 1000,
         'n_neuron': {'in': 7, 'hid': 1000, 'fb': 10},
         'lambda': 1.5,
         'learning_rate': 0.01,
         'tau': (10, 2),
         'n_neuron_per_tau': (700, 300),
         'n_neuron_of_readout': {'l1': 100, 'l2': 40, 'l3': 10, 'top': 3},
         'reward': 0.8,
         'penalty_for_wallcrash': -0.1,
         'penalty_for0_non-switchflag': -0.8,
         'smoothing_factor': 0.1,
         'discount_rate': 0.9,
         'connect_rate': 10,
         'actor_limit': 0.8,
         }

    def __init__(self):
        self.dirs = director.DirectoryProcessing()
        self.dirs.root_name = 'data'
        self.f = open(f'{self.dirs.root_name}para', 'a')
        self.f.write(f'{str(datetime.now())}"\n"')
        for key, value in self.p.items():
            self.f.write(f'{k} = {str(value)}"\n"')
        self.f.close()


class Module:
    def __init__(self):
        para = Database.p

    def


def main():
    para = Database().p
    np.random.seed(para['seed'])

    # print(para['episode'])
    for ep in range(para['episode']):
        print(f'epi: {ep}')


if __name__ == "__main__":
    main()
