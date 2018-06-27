#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as xp
import sys

import Neuron
seed_val = 1
xp.random.seed(seed_val)
np.random.seed(seed_val)

neuron = Neuron()

class Reservoir(neuron):
    def __init__(self):
        super(Reservoir, self).__init__()
