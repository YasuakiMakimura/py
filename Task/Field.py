import numpy as np
from math import sqrt
from itertools import combinations
from typing import Tuple
import pygame
from pygame.locals import *
import pygame.mixer
from Detection import circle_hit_circle
import object_draw


# class SetObjects:
#     def __init__(self, *objects):
#         self.objs = objects
#
#     def __call__(self, n_check):
#         def sum_radius(obj1_radius, obj2_radius):
#             return obj1_radius + obj2_radius
#
#         def distance(obj1_position, obj2_position):
#             obj1_p = np.array(obj1_position)
#             obj2_p = np.array(obj2_position)
#             return sqrt(np.sum((obj1_p - obj2_p) ** 2))
#
#         def outerwall_distance(_1, _2):
#             d = distance(_1.p, _2.p)
#             sum_r = sum_radius(_1.r, _2.r)
#             return d - sum_r
#
#         def reset_position(objs):
#             for obj in objs:
#                 # flag = obj._positionFlag
#                 if obj._positionFlag == 'random':
#                     obj.p = 'random'
#                 elif obj._positionFlag == 'fix':
#                     pass
#
#         def reset_count_check():
#             if n_check == 0:
#                 raise MyException(f"because a number of times to reset obj.p "
#                                   f"became more than limit of count, "
#                                   f"obj.p couldn't be decided.")
#
#         def fixposition_crash_check(_1, _2):
#             if _1._positionFlag == 'fix':
#                 if _2._positionFlag == 'fix':
#                     raise MyException(f'Crash between objects which the position is fixed ')
#
#         crash = False
#         for obj1, obj2 in combinations(self.objs, 2):
#             if outerwall_distance(obj1, obj2) <= 0:
#                 fixposition_crash_check(obj1, obj2)
#                 reset_count_check()  # ここでリミットを迎える
#                 crash = True
#                 reset_position(self.objs)
#                 break
#         if crash:
#             return self.position_check(n_check - 1)


class Field:
    rgb = (255, 255, 255)

    def __init__(self, size=400):
        pygame.init()
        pygame.key.set_repeat(5, 5)
        self.surface = pygame.display.set_mode((size, size))
        self.surface.fill(self.rgb)
        self.clock = pygame.time.Clock()
        self.update = pygame.display.update

    def draw_objects(self, *objects):
        self.surface.fill(self.rgb)
        for obj in objects:
            pygame.draw.circle(self.surface, obj.c, obj.p, obj.r, 2)

    def draw_agent_track(self, rgb: Tuple[int, int, int], pointlist):
        pygame.draw.lines(self.surface, rgb, False, pointlist)

    @staticmethod
    def objects_place(num_replace, *objects):

        def all_objects_replace(objs):
            for obj in objs:
                if obj() == 'random':
                    obj.p = 'random'

        crash = False
        for i in range(num_replace):
            for obj_1, obj_2 in combinations(objects, 2):
                if circle_hit_circle(obj_1.p[0], obj_1.p[1], obj_1.r,
                                     obj_2.p[0], obj_2.p[1], obj_2.r):
                    if obj_1() is obj_2() is 'fix':
                        raise ValueError('Crash between objects which the position is fixed')
                    crash = True
                    all_objects_replace(objects)
                    break

            if crash:
                crash = False
                continue

            else:
                return

        raise ValueError("because a number of times to reset obj.p "
                         "became more than limit of count, "
                         "obj.p couldn't be decided.")

    # @staticmethod
    # def objects_distance(obj1_position, obj2_position):
    #     obj1_p = np.array(obj1_position)
    #     obj2_p = np.array(obj2_position)
    #     return sqrt(np.sum((obj1_p - obj2_p) ** 2))
    #
    # def objects_angle(self, agent_position, obj2_position):
    #     distance = self.objects_distance(agent_position, obj2_position)
    #     cos = (obj2_position[0] - agent_position[0])/(distance + 1)
    #     sin = (obj2_position[1] - agent_position[1])/(distance + 1)
    #     return sin, cos
    #
    # def wall_distance(self, agent_position):
    #     right_wall = self.size - agent_position[0]
    #     left_wall = agent_position[0]
    #     up_wall = self.size - agent_position[1]
    #     down_wall = agent_position[1]
    #     return right_wall, left_wall, up_wall, down_wall


if __name__ == '__main__':
    pass
