# coding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *
import itertools as it
from copy import *
import functools
from collections import namedtuple, defaultdict
import pandas as pd
import pickle
from typing import Tuple, List, Iterable, Union, Optional
import cython
from time import perf_counter
# import pyximport
# pyximport.install()


if __name__ == "__main__":
    import Neu_Reser as neu
    import FunctionGenerator_ReNN as fg
    import RealTimePlot_ReNN as rtp

    np.random.seed(1)
    EPISODE = 10000
    # noinspection PyTypeChecker
    rv = neu.Reservoir(2, 10000, 1, v_tau=(2, 1, 10), n_tau=(5000, 4000, 1000))
    fg = fg.FG()
    rtp = rtp.RTP(Num_steps=EPISODE, Num_fig=4, figsize=(10, 5))
    fg.set_wave(Hz=1)
    rtp.init_sub_graph(0, "target")
    start = perf_counter()
    for ep in range(EPISODE):
        sys.stdout.write("\n{:d}".format(ep))
        # rv.ru_comp((rv.ltau, rv.stau), (rv.num / 2, rv.num / 2), rv_in=rv.in_o, rec_in=rv.old_ro, fb_in=rv.read_o)
        rv.ru_comp()
        rv.ro_comp()
        rv.read_o_comp()
        t = fg.gene_wave("wave1", ep)
        rtp.update_all([rv.read_o[0], rv.ro[4], rv.ro[5], rv.ro[9]])
        rtp.update_sub_graph(t, "target")
        if ep % 1000 == 0:
            rtp.plot()
    plt.show()
    end = perf_counter()
    print(f'time of compile: {end - start}')
