import numpy as np
from typing import Union
from math import sqrt


# Object super class
class Object:
    _r = None
    _p = None

    # Constructor
    def __init__(self, field_size, c, r=None, p='random'):
        # Params
        self.r = r
        self.c = c
        self.field_size = field_size
        if p == 'random':
            self.pos_type = 'random'
        else:
            self.pos_type = 'fix'
            self.p = p

    # end __init__

    ###################################
    # PROPERTIES
    ###################################

    @property
    def r(self):
        return self._r

    @property
    def p(self):
        return self._p

    ###################################
    # SETTERS
    ###################################

    @r.setter
    def r(self, radius):
        if radius < 0:
            raise ValueError(f'Radius of Object must be 0 and over')
        if radius is None:
            radius = 0.0
        self._r = int(radius)

    @p.setter
    def p(self, xy: Union[tuple, list, str]):
        low, high = self.r, self.field_size - self.r
        if isinstance(xy, str):
            if xy == 'random':
                self.pos_type = 'random'
                self._p = np.random.randint(low, high, (2,))
            else:
                raise TypeError(f'For obj.p, be not substituted excepting "random", '
                                f'tuple or list of position for a object')

        elif isinstance(xy, (list, tuple, np.ndarray)):
            if xy[0] < low or xy[0] > high or xy[1] < low or xy[1] > high:
                    raise ValueError(f'Object.p set up in the out of a field')

            self.pos_type = 'fix'
            self._p = np.array([int(xy[0]), int(xy[1])])

        else:
            raise NotImplementedError(f'Non-expected error')

    ##########################################
    # PUBLIC
    ##########################################

    # end Object class


# Switch object
class Switch(Object):

    def __init__(self, field_size, c, r=None, p='random'):
        super().__init__(field_size, c, r, p)

        # Params
        self.input_flag = 0.0
        self.push_flag = False

    # end __init__

    ###########################################
    # PROPERTIES
    ###########################################

    ###########################################
    # PUBLIC
    ###########################################

    def init_flag(self):
        self.input_flag = 0.0
        self.push_flag = False

    def check(self, target_p, switch_input):
        if sqrt(np.sum((self.p - target_p) ** 2.0)) < self.r:
            self.push_flag = True
            self.input_flag = switch_input
        else:
            self.input_flag = 0

    # end Switch class


# Goal object
class Goal(Object):

    def __init__(self, field_size, c, r=None, p='random'):
        super().__init__(field_size, c, r, p)

        # Params
        self.push_flag = False

    # end __init__

    ###########################################
    # PROPERTIES
    ###########################################

    ###########################################
    # PUBLIC
    ###########################################

    def init_flag(self):
        self.push_flag = False

    def push(self, target_p):
        if sqrt(np.sum((self.p - target_p) ** 2.0)) < self.r:
            self.push_flag = True
            return True
        else:
            return False

    # end Goal class


# Agent object
class Agent(Object):

    def __init__(self, field_size, c, r=None, p='random'):
        super().__init__(field_size, c, r, p)

        # Params
        self.wall_hit = False

    # end __init__

    ###########################################
    # PROPERTIES
    ###########################################

    ###########################################
    # PUBLIC
    ###########################################

    def move(self, x, y):
        old_pos = self.p
        new_pos = self.p + np.array([int(x), int(y)])
        self._p = self.wall_check(old_pos, new_pos)
        return self.p

    def wall_check(self, old_pos, new_pos):
        self.wall_hit = False
        low = self.r
        high = self.field_size - self.r
        old_p = old_pos
        new_p = new_pos
        if new_p[0] <= low or new_p[0] >= high:
            self.wall_hit = True
            new_p[0] = old_p[0]
        if new_p[1] <= low or new_p[1] >= high:
            self.wall_hit = True
            new_p[1] = old_p[1]
        return new_p

    # end Agent class


# Obstacle object
class Obstacle(Object):

    def __init__(self, field_size, c, r=None, p='random'):
        super().__init__(field_size, c, r, p)

