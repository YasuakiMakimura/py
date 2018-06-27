# coding=utf-8

import sys, os
sys.path.append(os.pardir)
import pickle_v0 as pic
import moving_average as m_a
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = '/home/maki/MEGA/pycharm_project/research_related/my_pro/data/empha4/exp_component/'
    expx = pic.r(f'{path}expx')
    expy = pic.r(f'{path}expy')
    m_expx = m_a.ma(expx, len_run=200000)
    m_expy = m_a.ma(expy, len_run=200000)
    # plt.plot(expx, label='Actor x origin', c='b')
    plt.plot(m_expx, label='Actor x', c='r')
    # plt.plot(expx, label='Actor x origin', c='b')
    plt.plot(m_expy, label='Actor y', c='b')
    plt.title('Exploration component')
    plt.legend()
    # plt.show()
    plt.savefig(f'{path}exp')
