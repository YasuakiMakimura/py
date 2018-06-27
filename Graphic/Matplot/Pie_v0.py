# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Union


class Pie:
    __c = None
    _label = None

    def __init__(self, num_fig: Union['str', 'int'], figsize=(9, 5), fontsize=10):
        plt.figure(num=num_fig, figsize=figsize, dpi=100)
        plt.style.use('ggplot')
        plt.rcParams['font.size'] = fontsize
        plt.axis('equal')  # jupyter等で楕円にならないようになる
        plt.subplots_adjust(left=0, right=0.7)

    # noinspection PyShadowingNames
    def out(self, data):
        plt.pie(data, colors=self.__c, labels=self._label, counterclock=False, startangle=90,
                autopct=lambda p: '{:.1f}%'.format(p) if p >= 5 else '')
        self.legend()

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, c):
        self.__c = c

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, lb):
        self._label = lb

    def legend(self):
        print(f'label in legend(): {self._label}')
        plt.legend(self._label, fancybox=True, loc='center left', bbox_to_anchor=(0.9, 0.5),
                   prop={'size': 12})

    def save(self, name):
        plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
        plt.pause(0.01)
        """
        bbox_inches:
            Bbox in inches. Only the given portion of the figure is saved. 
            If ‘tight’, try to figure out the tight bbox of the figure. 
            If None, use savefig.bbox
            
        pad_inches:
            Amount of padding around the figure when bbox_inches is ‘tight’. 
            If None, use savefig.pad_inches
        """


if __name__ == "__main__":
    data = [1011, 530, 355]
    p = Pie(num_fig='test')
    p.label = ['hoge', 'fuga', 'piyo']
    p.c = ['red', 'magenta', 'blue']
    p.out(data)
    p.save('test')

    # plt.show()
    plt.pause(3)
