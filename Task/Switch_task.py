import numpy as np
from ObjectSource import Switch, Goal, Agent
from Field import Field
from random import seed, randint

seed(1)
np.random.seed(1)


# Switch task
class Task:

    # Constructor
    def __init__(self):
        field_size = 400
        self.field_size = 400
        self.switch = Switch(field_size, c=(0, 255, 0), r=field_size * 0.1,
                             p='random')
        self.goal = Goal(field_size, c=(0, 0, 255), r=field_size * 0.1,
                         p='random')
        self.agent = Agent(field_size, c=(255, 0, 0), r=field_size * 0.05,
                           p='random')
        self.field = Field(size=field_size)

    # end __init__

    def reward_check(self, switch_input, reward,
                     direct_goal_penalty, wall_penalty):

        self.switch.check(target_p=self.agent.p, switch_input=switch_input)

        if self.goal.push(target_p=self.agent.p):
            if self.switch.push_flag:
                return reward
            else:
                return direct_goal_penalty

        elif self.agent.wall_hit:
            return wall_penalty

        else:
            return 0.0

    def display(self, frame_rate, pos_log, *objects):
        self.field.draw_objects(objects)
        self.field.draw_agent_track((0, 0, 0), pos_log)
        self.field.update(frame_rate)

    # end Task class


if __name__ == '__main__':
    task = Task()
    move_speed = 80
    p_log = []
    for epi in range(100000):
        print(f'episode: {epi}')
        task.switch.init_flag()
        task.goal.init_flag()
        task.field(100, task.agent, task.switch, task.goal)
        p_log.append(task.agent.p)

        for step in range(20):
            print(f'step:{step}')
            x_motion = randint(-move_speed, move_speed)
            y_motion = randint(-move_speed, move_speed)
            task.agent.move(x=x_motion, y=y_motion)
            reward \
                = task.reward_check(switch_input=10, reward=0.8,
                                    direct_goal_penalty=-0.8,
                                    wall_penalty=-0.1)

            p_log.append(task.agent.p)

            task.display(30, p_log, task.agent, task.switch, task.goal)

            if task.switch.push_flag:
                print('Switch ON!!!')
                print(task.switch.input_flag)

            if task.goal.push_flag:
                if task.switch.push_flag:
                    print('GOAL!!!')
                    input()
                else:
                    print('FAIL!!!')
                    input()
                break

        p_log = []
