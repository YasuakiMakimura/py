import numpy as np
import matplotlib.pyplot as plt
from random import randint, seed
# import pygame
from typing import Union, Tuple


class Object:
    def __init__(self, field_size):
        self.__r = None
        self.__p = np.array([0, 0])
        self.__c = (0, 0, 0)
        self.__fieldsize = field_size
        self.random_pos = None

    @property
    def r(self):
        return self.__r

    @property
    def p(self):
        return self.__p

    @property
    def c(self):
        return self.__c

    @property
    def field_size(self):
        return self.__fieldsize

    @r.setter
    def r(self, radius):
        if radius < 0:
            raise ValueError(f'Radius of Object must be 0 and over')
        self.__r = int(radius)

    @p.setter
    def p(self, xy: Union[tuple, list, str]):
        low, high = self.r, self.field_size - self.r
        if isinstance(xy, str):
            if xy == 'random':
                self.random_pos = True
                self.__p = np.random.randint(low, high, (2,))
            else:
                raise TypeError(f'For obj.p, be not substituted excepting "random", tuple or'
                                  f'list of position for a object')
        elif isinstance(xy, (list, tuple, np.ndarray)):
            if xy[0] < low or xy[0] > high or xy[1] < low or xy[1] > high:
                    raise ValueError(f'Object.p set up in the out of a field')
            self.random_pos = False
            self.__p = np.array([int(xy[0]), int(xy[1])])
        else:
            raise Exception(f'Non-expected error')

    @c.setter
    def c(self, color_rgb: Tuple[int, int, int]):
        self.__c = color_rgb

    @field_size.setter
    def field_size(self, field_size):
        self.__fieldsize = field_size


class Switch(Object):
    def __init__(self, field_size):
        super().__init__(field_size)
        self._input_flag = None
        self._push = False

    @property
    def input_flag(self):
        return self._input_flag
    
    @input_flag.setter
    def input_flag(self, input_flag):
        self._input_flag = input_flag

    @property
    def push(self):
        return self._push

    @push.setter
    def push(self, push_flag: bool):
        self._push = push_flag


class Goal(Object):
    def __init__(self, field_size):
        super().__init__(field_size)
        self._push = False

    @property
    def push(self):
        return self._push

    @push.setter
    def push(self, push_flag: bool):
        self._push = push_flag


class Agent(Object):
    def __init__(self, field_size):
        super().__init__(field_size)

    def move(self, x, y):
        return self.p + np.array([int(x), int(y)])

    def wall_check(self, old_pos, new_pos, wall_penalty):
        low = self.r
        high = self.field_size - self.r
        w_pena = 0.0
        old_p = old_pos
        new_p = new_pos
        if new_p[0] <= low or new_p[0] >= high:
            w_pena = wall_penalty
            new_p[0] = old_p[0]
        if new_p[1] <= low or new_p[1] >= high:
            w_pena = wall_penalty
            new_p[1] = old_p[1]
        return new_p, w_pena

