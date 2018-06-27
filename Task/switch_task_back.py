#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import math
import sys
import pygame.mixer
from random import randint, seed
from math import sin, cos, radians
import numpy as np
from itertools import combinations
from typing import Union, Tuple

seed(1)

# todo Noneにしておいて間違ってmain内でFieldクラスのインスタンスを後にしてしまったときでもエラー処理できるようにする
field_size = 200

# todo タスクでよく使いそうな関数を他のモジュールにする？そしてそれをimportして使う？
def distance(obj1_position, obj2_position):
    obj1_p = np.array(obj1_position)
    obj2_p = np.array(obj2_position)
    return math.sqrt(np.sum((obj1_p - obj2_p) ** 2))


class MyException(Exception):
    pass


class Object:
    __r = None
    __p = np.array([0, 0])
    __c = (0, 0, 0)
    _positionFlag = None

    @property
    def r(self):
        return self.__r

    @r.setter
    def r(self, radius):
        if radius < 0:
            raise MyException(f'Radius of Object must be 0 and over')
        self.__r = int(radius)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, xy: Union[tuple, list, str]):
        low, high = self.r, field_size - self.r
        if isinstance(xy, str):
            if xy == 'random':
                self._positionFlag = 'random'
                self.__p = randint(low, high), randint(low, high)
            else:
                raise MyException(f'For obj.p, be not substituted excepting "random", tuple or'
                                  f'list of position for a object')
        elif isinstance(xy, (list, tuple)):
            if xy[0] < low or xy[0] > high:
                if xy[1] < low or xy[1] > high:
                    # print(f'low: {low}, high: {high}')
                    # print(np.array(xy))
                    raise MyException(f'Object.p set up in the out of a field')
            self._positionFlag = 'fix'
            self.__p = int(xy[0]), int(xy[1])
        else:
            raise MyException(f'Non-expected error')

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, c_rgb: Tuple[int, int, int]):
        self.__c = c_rgb

    # def setp(self, field_size, mode='random'):
    #     if mode == 'random':
    #         return np.random.uniform(-field_size, field_size, (2,))
    #     elif mode == 'fix':

    # def move(self, xy: Union[tuple, list]):  #     self.p[0] += xy[0]  #     self.p[1] += xy[1]


class Switch(Object):
    limit_flag = 10
    __vflag = None

    def __init__(self, switch_input):
        self.set_flag = switch_input

    @property
    def current_flag(self):
        return self.__vflag

    @current_flag.setter
    def current_flag(self, value):
        if self.__vflag is None:
            raise MyException(f'Switch Flag is None, must be int or float')
        self.__vflag = value


class Goal(Object):
    __rew = None

    @property
    def rew(self):
        return self.__rew

    @rew.setter
    def rew(self, reward):
        if self.__rew is None:
            raise MyException(f"Reward is not set up with int or float")
        self.__rew = reward


class Agent(Object):
    # todo エージェントの軌道を描けるようにする
    # todo ここはいちいちndarray型になおして、またtuple型に直しているので煩雑
    __sw = None
    __gl = None
    __rew = None
    __pena = None
    # todo スイッチが複数あるときも対応したプログラムになっているか
    __switchflag = None
    def __init__(self, switch, goal):
        self.sw = switch
        self.gl = goal

    # todo やっぱりswitch, goalのインスタンスはエージェントに引数として与えたほうがいいかな？
    # @property
    # def target_switch(self):
    #     return self.__sw
    #
    # @target_switch.setter
    # def target_switch(self, switch_instance):
    #     self.__sw = switch_instance
    #
    # @property
    # def target_goal(self):
    #     return self.__gl
    #
    # @target_goal.setter
    # def target_goal(self, goal_instance):
    #     self.__gl = goal_instance

    @property
    def reward(self):
        return self.__rew

    @reward.setter
    def reward(self, reward):
        self.__rew = reward

    @property
    def penalty(self):
        return self.__pena

    @penalty.setter
    def penalty(self, penalty):
        self.__pena = penalty

    def move(self, xy: Union[list, tuple]):
        # print(f'ptype: {type(self.p)}')
        p = np.array(self.p) + np.array(xy)
        self.p = tuple(self.wallcheck(self.p, p))
        if switchflag_check() == 'in':

        if self.switch_check() == 'in':
            # todo ここから
            # self.target_switch.flag =

    def wallcheck(self, old_position: Union[list, tuple], new_position: Union[list, tuple]):
        # self.wall_flag = 'off'
        low = self.r
        high = field_size - self.r
        old_p = old_position
        new_p = new_position
        if new_p[0] <= low:
            # self.wall_flag = 'on'
            new_p[0] = old_p[0]
        if new_p[0] >= high:
            # self.wall_flag = 'on'
            new_p[0] = old_p[0]
        if new_p[1] <= low:
            # self.wall_flag = 'on'
            new_p[1] = old_p[1]
        if new_p[1] >= high:
            # self.wall_flag = 'on'
            new_p[1] = old_p[1]
        return new_p

    def switch_check(self):# todo とりあえずスイッチの踏む順番関係なしに、全部踏めばいい場合で書いている。
        for i, sw in enumerate(self.sw):
            if distance(self.p, sw.p) <= sw.r:
                sw.current_flag = sw.set_flag
                yield 'in'


        # todo エージェントがスイッチに入っているかどうかチェックし、入っていた場合には'in'を返す
        # todo　入力フラグ自体はmove method内で代入する

    def goal_check(self):
        # for i, gl in enumerate(self.gl):# todoゴール複数のときの場合の for文
        if distance(self.p, self.gl.p) <= self.gl.r:
            yield 'goal'
        # else:
        #     return 'out'
        # todo エージェントがゴールに入っているかどうかチェックし、入っていた場合には'in'を返す
        # todo　報酬、罰自体はmove method内で代入する

    def mission_check(self):
        if self.goal_check() == 'goal':
             self.switchflag_check() == :
                return 'mission clear'





        ...


class SetObjects:
    def __init__(self, *objects):
        self.objs = objects

    def position_check(self, n_check):
        def sum_radius(obj1_radius, obj2_radius):
            return obj1_radius + obj2_radius

        def outerwall_distance(_1, _2):
            d = distance(_1.p, _2.p)
            sum_r = sum_radius(_1.r, _2.r)
            return d - sum_r

        def reset_position(objs):
            for obj in objs:
                # flag = obj._positionFlag
                if obj._positionFlag == 'random':
                    obj.p = 'random'
                elif obj._positionFlag == 'fix':
                    pass

        def reset_count_check():
            if n_check == 0:
                raise MyException(f"because a number of times to reset obj.p "
                                  f"became more than limit of count, "
                                  f"obj.p couldn't be decided.")

        def fixposition_crash_check(_1, _2):
            if _1._positionFlag == 'fix':
                if _2._positionFlag == 'fix':
                    raise MyException(f'Crash between objects which the position is fixed ')

        crash = False
        for obj1, obj2 in combinations(self.objs, 2):
            if outerwall_distance(obj1, obj2) <= 0:
                fixposition_crash_check(obj1, obj2)
                reset_count_check()  # ここでリミットを迎える
                crash = True
                reset_position(self.objs)
                break
        if crash:
            # print(n_check)
            return self.position_check(n_check - 1)


class Field:
    __rgb = (255, 255, 255)

    def __init__(self, size):
        global field_size
        field_size = size
        pygame.init()
        pygame.key.set_repeat(5, 5)
        self.surface = pygame.display.set_mode((size, size))
        self.surface.fill(self.rgb)
        self.fpsclock = pygame.time.Clock()  # フレームレート(fps)制御ができる
        # self.sw = Switch()
        # self.gl = Goal()
        # self.ag = Object()
        # self.sw.p = field_size * 0.8, field_size * 0.8
        # self.gl.p = field_size * 0.2, field_size * 0.2
        # self.ag.p = field_size * 0.5, field_size * 0.5
        # self.cir = pygame.draw.circle
        self.display = pygame.display.update

    @property
    def rgb(self):
        return self.__rgb

    @rgb.setter
    def rgb(self, rgb):
        if not isinstance(rgb, (tuple, list)):
            raise MyException(f'rgb variable must be tuple or list')
        self.__rgb = rgb

    @property
    def size(self):
        return field_size

    # @property
    # def update(self):
    #     return pygame.display.update()

    def draw_circle(self, *objects):
        self.surface.fill(self.rgb)
        for obj in objects:
            pygame.draw.circle(self.surface, obj.c, obj.p, obj.r)

        # self.cir(self.surface, (0, 0, 255), self.gl.p, self.gl.r)  # self.cir(self.surface, (0, 255, 0), self.sw.p, self.sw.r)  # self.cir(self.surface, (255, 0, 0), self.ag.p, self.ag.r)


def main():
    # Field classのインスタンス形成
    field = Field(size=200)
    # switch, goal, agentのインスタンス形成
    sw = Switch()
    gl = Goal()
    # todo switch, goalのインスタンスはエージェントのクラスに引数として与える。与えたswitch, goalがそのエージェントのターゲットとなる。
    ag1 = Agent()
    ag2 = Agent()
    # switch, goal, agentの半径を決定
    sw.r = field.size * 0.1
    gl.r = field.size * 0.1
    ag1.r = field.size * 0.1
    ag2.r = field.size * 0.1
    # switch, goal, agentの色を決定
    sw.c = (0, 255, 0)
    gl.c = (0, 0, 255)
    ag1.c = (255, 0, 0)
    ag2.c = (255, 0, 0)
    # switch, goal, agentの重なりの検査
    setobj = SetObjects(sw, gl, ag1)
    # setobj = SetObjects(sw, gl, ag1, ag2)

    # Simulation fieldを形成

    step = 0
    move_value = 8

    for epi in range(100000):
        print(f'episode: {epi}')
        # switch, goal, agentのポジションを決定
        sw.p = field.size / 2, field.size / 1.2
        gl.p = 'random'
        # if epi == 10:
        #     gl.p = field.size / 2, field.size / 2
        ag1.p = 'random'
        # ag2.p = 'random'
        # switch, goal, agentの重なりをチェックし、ポジションを再配置
        setobj.position_check(100)
        for step in range(200):
            x_motion = randint(-move_value, move_value)
            y_motion = randint(-move_value, move_value)
            # oldp = ag1.p
            ag1.move([x_motion, y_motion])
            if ag1.
            # newp = ag1.p
            # if ag1.wall_flag == 'on':
            #     print('ON!')
            #     print(f'oldp: {oldp}, newp: {newp}')
            #     input()
            field.draw_circle(sw, gl, ag1)
            field.display()
            field.fpsclock.tick(100)
            # input()
        # print('end')
        # field.draw_circle(sw, gl, ag)
        # field.display()
        # field.fpsclock.tick(30)
        # ここからはキーイベントのコード
        # input('keydown  please')
        # print('flag')
        # for ev in pygame.event.get():
        #     if ev.type == QUIT:
        #         pygame.quit()
        #         sys.exit()
        #     elif ev.type == KEYDOWN:
        #         step += 1
        #         print(f'step: {step}')
        #         if ev.key == K_LEFT:
        #             ag.move((-move_value, 0))
        #         if ev.key == K_RIGHT:
        #             ag.move((move_value, 0))
        #         if ev.key == K_UP:
        #             ag.move((0, -move_value))
        #         if ev.key == K_DOWN:
        #             ag.move((0, move_value))
        #         if ev.key == K_ESCAPE:
        #             pygame.quit()
        #             sys.exit()
        # print('flag2')
        # field.draw_circle(sw, gl, ag1)
        # field.draw_circle(sw, gl, ag1, ag2)


if __name__ == '__main__':
    main()
