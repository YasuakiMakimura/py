# coding=utf-8

import numpy as np
from copy import deepcopy, copy
from bokeh.plotting import figure, save
from bokeh.io import output_notebook, output_file
from bokeh.io import show


class Line:
    __plotflag = False

    def __init__(self, n_line=1, xlabel='X', ylabel='Y'):
        """
        :param n_line: プロットするグラフの数
        :type n_line: int
        self.figure: プロットする図
        self.hanger: データを格納するりスト
        """
        # self.init_figure = figure(plot_width=400, plot_height=400,
        #                           x_axis_label=xlabel,
        #                           y_axis_label=ylabel)

        self.figure = figure(plot_width=400, plot_height=400,
                             x_axis_label=xlabel, y_axis_label=ylabel)
        self.n_line = n_line
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._legend = [None for i in range(self.n_line)]
        self._line_color = [tuple(np.random.randint(0, 200, (3,))) for i in range(self.n_line)]
        if self.n_line == 1:
            self.init_hanger = []
            self.hanger = []
            self.x = []
        else:
            self.init_hanger = [[] for i in range(n_line)]
            self.hanger = [[] for i in range(n_line)]
            self.x = [[] for i in range(n_line)]

    @property
    def legend(self):
        return self._legend

    @legend.setter
    def legend(self, name):
        if len(name) != self.n_line:
            raise AttributeError("Number of line don't match with len of legend list")
        self._legend = name

    @property
    def line_color(self):
        return self._line_color

    @line_color.setter
    def line_color(self, *color):
        if len(color) != self.n_line:
            raise AttributeError("Number of line don't match with len of line color list")
        self._line_color = color

    # @property
    # def x(self):
    #     for offset, data in enumerate(self.hanger):
    #         self._x.append(range(len(data)))
    #     return self._x
    #
    # @x.setter
    # def x(self, x):
    #     self._x = x

    # def store(self, data, offset=None):
    #     """
    #     データを格納する関数
    #     :param offset:
    #     :param data: プロットするための元データ
    #     :type data: int, float
    #     self.n_line: グラフの数
    #     self.hanger: データを格納するリスト
    #     :return:
    #     """
    #     if isinstance(data, (tuple, list)):
    #         if len(data) != self.n_line:
    #             raise AttributeError("Number of line don't match with len(data)")
    #         self.hanger[offset].append(data)
    #
    #     elif isinstance(data, (int, float)):
    #         if self.n_line != 1:
    #             raise AttributeError("Number of line don't match with len(data)")
    #         self.hanger.append(data)
    #
    #     else:
    #         raise TypeError(f'"data" of store() must '
    #                         f'"tuple", "list", "int", or "float"')

    def store(self, data, offset=None):
        """
        データを格納する関数
        :param offset:
        :param data: プロットするための元データ
        :type data: int, float
        self.n_line: グラフの数
        self.hanger: データを格納するリスト
        :return:
        """

        if isinstance(data, (int, float)):
            self.hanger[offset].append(data)

        else:
            raise TypeError(f'"data" of store() must '
                            f'"tuple", "list", "int", or "float"')

    def set_x(self, x=None, offset=None):
        if x is offset is None:
            for offset, data in enumerate(self.hanger):
                self.x[offset] = range(len(data))
        elif x is None:
            self.x[offset] = range(len(self.hanger[offset]))
        else:
            self.x[offset] = x

    def plot(self, store_initialization=True):
        if self.n_line == 1:
            self.figure.line(self.x, self.hanger, line_color=(0, 0, 0))
        else:
            if len(self.hanger) != self.n_line:
                raise AttributeError("Number of line don't match with len(data)")
            for offset in range(self.n_line):
                self.figure.line(self.x[offset], self.hanger[offset],
                                 legend=self.legend[offset],
                                 line_color=self.line_color[offset])
        self.__plotflag = True
        if store_initialization:
            self.hanger = deepcopy(self.init_hanger)

    def save(self, filename: str):
        if self.__plotflag:
            output_file(f'{filename}.html')
            save(self.figure, filename=f'{filename}.html')
            self.__plotflag = False
            self.figure =  figure(plot_width=400, plot_height=400,
                                  x_axis_label=self.xlabel,
                                  y_axis_label=self.ylabel)
        else:
            raise AttributeError(f'save() be called after plot()')


if __name__ == "__main__":
    import random
    boke = Line(n_line=2)
    # boke.legend =''
    for i in range(20):
        rnd = random.uniform(-1, 1)
        rnd_2 = random.uniform(-1, 1)
        # boke.store(rnd)
        boke.store((rnd, rnd_2))
    boke.plot()
    show(boke.figure)
    boke.save('test')
