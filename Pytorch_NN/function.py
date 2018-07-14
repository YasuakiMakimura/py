# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import torch


# Generate normal random number
def n_rand(mean=0.0, std=1.0):
    def gene(size):
        n_r = np.random.normal(mean, std, size)
        return torch.FloatTensor(n_r)
    return gene


# Generate uniform random number
def u_rand(low=-1.0, high=1.0):
    def gene(size):
        u_r = np.random.uniform(low, high, size)
        return torch.FloatTensor(u_r)
    return gene


# Wrapper of n_rand()
def generate_nw(mean=0.0, std=1.0):
    return n_rand(mean, std)


# Wrapper of u_rand()
def generate_uw(low=-1.0, high=1.0):
    return u_rand(low, high)
