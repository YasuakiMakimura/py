# -*- coding:utf-8 -*- 

import numpy as np
import sys
import matplotlib.pyplot as plt
from pylab import *
import seaborn
import itertools as it
from copy import *
import functools


def min_max(self, x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min)/(max - min)
    return result




if __name__ == "__main__":
    pass
