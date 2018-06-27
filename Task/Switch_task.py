import numpy as np
from math import sqrt
from ObjectSource import Switch, Goal, Agent
from Field_kai import Field
from random import seed, randint
import pygame
from pygame.locals import *
import sys

seed(1)
np.random.seed(1)


class Task:
    def __init__(self):
        field_size = 400
        self.switch = Switch(field_size)
        self.goal = Goal(field_size)
        self.agent = Agent(field_size)
        self.switch.r = field_size * 0.18
        self.goal.r = field_size * 0.18
        self.agent.r = field_size * 0.1
        self.switch.c = (0, 255, 0)
        self.goal.c = (0, 0, 255)
        self.agent.c = (255, 0, 0)
        self.field = Field(size=field_size)
        self.reward = 0
        self.agent_pos_log = []

    def agent_log_init(self):
        self.agent_pos_log = []
        self.agent_pos_log.append(self.agent.p)

    @property
    def agent_log(self):
        self.agent_pos_log.append(self.agent.p)
        return self.agent_pos_log

    def agent_move(self, x, y, switch_input, reward,
                   direct_goal_penalty, wall_penalty=0.0):
        """

        :param reward:
        :param direct_goal_penalty:
        :param wall_penalty:
        :param switch_input:
        :param x: xの移動量
        :param y: yの移動量
        :return: エージェントが移動する関数
        """
        old_pos = self.agent.p
        new_pos = self.agent.move(x, y)
        self.agent.p, self.reward\
            = self.agent.wall_check(old_pos, new_pos, wall_penalty)
        # self.field.draw_agent_track((0, 0, 0), tuple(old_pos), tuple(new_pos))
        if self.switch_push():
            self.switch.input_flag = switch_input
            self.switch.push = True
        else:
            self.switch.input_flag = 0
        if self.goal_push():
            self.goal.push = True
            if self.switch.push:
                self.reward = reward
            else:
                self.reward = direct_goal_penalty

    def switch_push(self):
        """
        エージェントがスイッチに入っているかをチェックする

        self.switch_input:
            スイッチ入力

        self.switch_pushFlag:
            初期値は'off'
            スイッチに一回でも入ったら'on'となる
        """
        if sqrt(np.sum((self.switch.p - self.agent.p) ** 2.0)) \
                < self.switch.r:
            return True
        else:
            return False

    def goal_push(self):
        """
        エージェントがゴールに入っているかをチェックする

        self.switch_pushFlag:
            初期値は'off'
            スイッチに一回でも入ったら'on'となる

        self.reward:
            報酬

        self.direct_goal_penalty:
            エージェントがスイッチに入らず、直接ゴールしてしまった場合の罰
        """
        if sqrt(np.sum((self.goal.p - self.agent.p) ** 2.0)) \
                < self.goal.r:
            return True
        else:
            return False

    def display(self, framerate):
        self.field.draw_objects(self.switch, self.goal, self.agent)
        self.field.draw_agent_track((0, 0, 0), self.agent_log)
        self.field.update()
        self.field.clock.tick(framerate)

    @property
    def objects_place(self):
        return self.field.objects_place

    # def distance_ag_gl(self):
    #     """
    #     :return:エージェントとゴールの距離
    #     """
    #     return self.field.objects_distance(self.agent.p, self.goal.p)
    #
    # def distance_ag_sw(self):
    #     """
    #     :return:エージェントとスイッチの距離
    #     """
    #     return self.field.objects_distance(self.agent.p, self.switch.p)
    #
    # def angle_ag_gl(self):
    #     """
    #     :return:sin, cos （エージェントとゴールの相対角度）
    #     """
    #     return self.field.objects_angle(self.agent.p, self.goal.p)
    #
    # def angle_ag_sw(self):
    #     """
    #     :return:sin, cos （エージェントとスイッチの相対角度）
    #     """
    #     return self.field.objects_angle(self.agent.p, self.switch.p)
    #
    # def wall_distance(self):
    #     """
    #     :return:right_wall, left_wall, up_wall, down_wall (エージェントと壁の距離)
    #     """
    #     return self.field.wall_distance(self.agent.p)
    #

    # def get_reward(self):
    #     """
    #     報酬または罰を返す
    #     :return: 報酬 or 罰 or 0
    #     """
    #     if self.agent.reward == self.reward:
    #         r = self.agent.reward
    #     elif self.agent.penalty == self.direct_goal_penalty:
    #         r = self.agent.penalty
    #     elif self.agent.penalty == self.wall_penalty:
    #         r = self.agent.penalty
    #     else:
    #         r = 0
    #     self.agent.reward = 0
    #     self.agent.penalty = 0
    #     return r


if __name__ == '__main__':
    task = Task()
    move_speed = 80
    for epi in range(100000):
        print(f'episode: {epi}')
        task.agent.p = 'random'
        task.switch.p = 'random'
        task.goal.p = 'random'
        task.switch.push = False
        task.goal.push = False
        task.objects_place(100, task.agent, task.switch, task.goal)
        task.agent_log_init()
        task.reward = 0
        # task.display(framerate=100)
        for step in range(20):
            print(f'step:{step}')
            x_motion = randint(-move_speed, move_speed)
            y_motion = randint(-move_speed, move_speed)
            task.agent_move(x=x_motion, y=y_motion, switch_input=10,
                            reward=0.8, direct_goal_penalty=-0.8,
                            wall_penalty=-0.1)
            task.display(framerate=30)
            if task.switch.push:
                print('Switch!!!')
            if task.goal.push:
                if task.switch.push:
                    print('GOAL!!!')
                    input()
                else:
                    print('FAIL!!!')
                    input()
                break
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)
