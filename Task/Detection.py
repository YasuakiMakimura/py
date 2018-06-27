# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


# 円と円の当たり判定
def circle_hit_circle(x0, y0, r0, x1, y1, r1):
    dx = x0 - x1
    dy = y0 - y1
    sum_r = r0 + r1
    return (dx ** 2 + dy ** 2) <= sum_r ** 2


# 円と円の侵入判定(円の中心が半径内に入っているかの判定)
def circle_enter_circle(x0, y0, x1, y1, victim_r1):
    dx = x0 - x1
    dy = y0 - y1
    return (dx ** 2 + dy ** 2) < victim_r1
