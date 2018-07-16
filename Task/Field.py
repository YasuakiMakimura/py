from itertools import combinations
from typing import Tuple
import pygame
import pygame.mixer
from Detection import circle_hit_circle


# Task field
class Field:
    rgb = (255, 255, 255)

    # Constructor
    def __init__(self, size=400):
        pygame.init()
        pygame.key.set_repeat(5, 5)

        # Params
        self.size = size
        self.surface = pygame.display.set_mode((self.size, self.size))
        self.surface.fill(self.rgb)
        self._clock = pygame.time.Clock()

    # end __init__

    ############################################
    # PROPERTIES
    ############################################

    ############################################
    # PUBLIC
    ############################################

    def update(self, frame_rate):
        """
        フィールドを更新する関数
        :param frame_rate:
        :return:
        """
        pygame.display.update()
        self._clock.tick(frame_rate)

    def draw_objects(self, objects):
        """
        オブジェクトを描く関数
        :param objects:
        :return:
        """
        self.surface.fill(self.rgb)
        for obj in objects:
            pygame.draw.circle(self.surface, obj.c, obj.p, obj.r, 2)

    def draw_agent_track(self, rgb: Tuple[int, int, int], pointlist):
        """
        エージェントの軌道を描く関数
        :param rgb:
        :param pointlist:
        :return:
        """
        pygame.draw.lines(self.surface, rgb, False, pointlist)

    #################################################
    # STATIC
    #################################################

    @staticmethod
    def __call__(num_replace, *objects):
        """
        フィールド上に各オブジェクトを配置する関数
        :param num_replace:
        :param objects:
        :return:
        """

        def objects_set(objs):
            for obj in objs:
                if obj.pos_type == 'random':
                    obj.p = 'random'

        objects_set(objects)
        crash = False
        for i in range(num_replace):
            for obj_1, obj_2 in combinations(objects, 2):
                if circle_hit_circle(obj_1.p[0], obj_1.p[1], obj_1.r,
                                     obj_2.p[0], obj_2.p[1], obj_2.r):
                    if obj_1.pos_type is obj_2.pos_type is 'fix':
                        raise ValueError('Crash between objects which the position is fixed')
                    crash = True
                    objects_set(objects)
                    break

            if crash:
                crash = False
                continue
            else:
                return

        raise ValueError("because a number of times to reset obj.p "
                         "became more than limit of count, "
                         "obj.p couldn't be decided.")
