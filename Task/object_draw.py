# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import pygame
from pygame.locals import *
import pygame.mixer


def circle(surface, c, pos, r, line_width=0):
    """
    pygame 円を描写する
    :param surface:描くフィールドのインスタンス
    :param c: 円の色
    :param pos: 円のポジション
    :param r: 円の半径
    :param line_width:円の枠線の幅、デフォルトは0だが幅を指定すると枠線だけの円(塗りつぶしなし)となる
    :return: pygame.draw.circle(surface, color, pos, r, line_width)
    """
    return pygame.draw.circle(surface, c, pos, r, line_width)


def rect(surface, c, rect_range, width=0):
    """
    pygame 四角形を描写する
    :param surface:
    :param c:
    :param rect_range:
    :param width:
    :return:
    """
    return pygame.draw.rect(surface, c, rect_range, width)

