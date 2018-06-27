# -*- coding:utf-8 -*- 

import numpy as np
import sys
import matplotlib.pyplot as plt
from pylab import *
import seaborn
import itertools as it
from copy import *
import functools
from collections import namedtuple, defaultdict
from typing import List, Tuple, Iterable, Union, Optional
import pandas as pd
from typing import Sequence

import Logger as log


def mean(data: Sequence):
    """

    :param data: Sequence of the data tring to verify
    :return: Average of the data
    """
    return sum(data) / len(data)

def dev(data: Sequence):
    """

    :param data: Sequence of the data tring to verify
    :return: Deviation of the data
    """
    diff = []
    for num in data:
        diff.append(num - mean(data))
    return diff

def variance(data: Sequence):
    """

    :param data:  Sequence of the data tring to verify
    :return: Variance of the data
    """
    diff = dev(data)
    squared_diff = []
    for d in diff:
        squared_diff.append(d ** 2)
    variance = sum(squared_diff) / len(data)
    return variance

def s_dev(data: Sequence):
    """

    :param data: Sequence of the data tring to verify
    :return: Standard Deviation of the data
    """
    return variance(data) ** 0.5

def c_v(data: Sequence):
    """

    :param data: Sequence of the data tring to verify
    :return: Coefficient of the data

        use in case of verifing the variance of dataset that the average differed.
    """
    return s_dev(data) / mean(data)


if __name__ == "__main__":
    pass
