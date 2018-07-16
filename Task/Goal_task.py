# coding=utf-8

from random import randint
from ObjectSource import Goal, Agent
from Field import Field


class Task:
    def __init__(self):
        field_size = 400
        self.goal = Goal(field_size, c=(0, 0, 255), r=field_size * 0.2)
        self.agent = Agent(field_size, c=(255, 0, 0), r=field_size * 0.1)
        self.field = Field(field_size)

    def reward_check(self, reward, wall_penalty):
        if self.goal.push(target_p=self.agent.p):
            return reward

        elif self.agent.wall_hit:
            return wall_penalty

        else:
            return 0.0

    def display(self, frame_rate, pos_log, *objects):
        self.field.draw_objects(objects)
        self.field.draw_agent_track((0, 0, 0), pos_log)
        self.field.update(frame_rate)


if __name__ == "__main__":
    task = Task()
    move_speed = 80
    p_log = []
    for epoch in range(1000):
        print(f'ep: {epoch}')
        task.goal.init_flag()
        task.agent.p = 'random'
        task.goal.p = 'random'
        task.field(100, task.agent, task.goal)

        p_log.append(task.agent.p)

        for time in range(20):
            print(f'time: {time}')
            x_motion = randint(-move_speed, move_speed)
            y_motion = randint(-move_speed, move_speed)
            task.agent.move(x=x_motion, y=y_motion)
            reward = task.reward_check(reward=0.8, wall_penalty=-0.1)

            p_log.append(task.agent.p)

            task.display(10, p_log, task.agent, task.goal)

            if task.goal.push_flag:
                print('GOAL!!!')
                break

        p_log = []
