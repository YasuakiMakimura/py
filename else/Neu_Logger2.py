# coding=utf-8

# from collections import defaultdict
from typing import Sequence, Tuple
from pylab import *

print(__name__)


# noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
class Log:
    def __new__(cls, n_line: int, num_fig: int, subplot: bool = False,
                **kwargs):  # todo kwargsのそれぞれにデフォ値と違う型のものが入った時のエラー処理はしなくて良い？
        """

        :param n_line:
        :param num_fig:
        :param subplot:
        :param kwargs:
                        alpha: float, Tuple[float, ...]
        :return:
        """
        cls.def_k = {'c': None, 'y_low': 0, 'y_high': 1.0, 'x_low': 0,
                     'x_high': 50, 'grid': False, 'xlabel': 'x', 'ylabel': 'y',
                     'figsize': (6, 4), 'origin_line': False, 'originc': None,
                     'any_line': False, 'iter_any': [0.0], 'anyc': None,
                     'inf': None, 'auto_xrange': True, 'legend': None,
                     'alpha': None, 'scatter': False}

        for key_kwg, v_kwg in kwargs.items():
            try:
                cls.def_k[key_kwg]
            except KeyError:
                print(f'List of **kwargs.keys(): {cls.def_k.keys()}\n')
                raise AttributeError(
                    f'arg("{key_kwg}") does not exist  in kwargs of class ("{cls.__name__}")')
            else:
                cls.def_k[key_kwg] = v_kwg
        return super().__new__(cls)

    def __init__(self, n_line: int, num_fig: int, subplot: bool = False,
                 **kwargs):
        self.k = self.def_k
        self.n_line = n_line
        self.fig = plt.figure(num=num_fig, figsize=self.k['figsize'])
        self.lines = [0] * n_line
        if n_line == 1:
            ax = self.fig.add_subplot(111)
            plt.axis([self.k['x_low'], self.k['x_high'], self.k['y_low'],
                      self.k['y_high']])
            self.lines[0], = ax.plot(range(1), range(1), c=self.k['c'])
            if self.k['origin_line']:
                origin_line = [0]
                origin_line[0], = ax.plot(range(self.k['x_high']),
                                          [0] * self.k['x_high'],
                                          c=self.k['originc'])
            if self.k['any_line']:
                any_line = [0]
                any_line[0], = ax.plot(range(self.k['x_high']),
                                       self.k['iter_any'], c=self.k['anyc'])
            plt.ylabel(self.k['ylabel'], fontsize=18)
            plt.xlabel(self.k['xlabel'], fontsize=18)
            plt.tight_layout()

        else:
            ax_list = []
            y_list = [[0] * self.k['x_high'] for j in range(n_line)]

            if subplot:
                for iii in range(n_line):
                    ax_list.append(self.fig.add_subplot(n_line, 1, iii + 1))
                    ax_list[iii].set_ylim([self.k['y_low'], self.k['y_high']])
                    self.lines[iii], = ax_list[iii].plot(
                        range(self.k['x_high']), y_list[iii], c=self.k['c'])
                ax_list[int(n_line / 2)].set_ylabel(self.k['ylabel'],
                                                    fontsize=18)
                plt.xlabel(self.k['xlabel'], fontsize=18)
            else:
                plt.ylabel(self.k['ylabel'], fontsize=18)
                plt.xlabel(self.k['xlabel'], fontsize=18)

                for iii in range(n_line):
                    ax = self.fig.add_subplot(111)
                    self.lines[iii], = ax.plot(range(self.k['x_high']),
                                               y_list[iii], c=self.k['c'][iii])
                    if self.k['origin_line']:
                        origin_line = [0]
                        origin_line[0], = ax.plot(range(self.k['x_high']),
                                                  [0] * self.k['x_high'],
                                                  c=self.k['originc'])
                    if self.k['any_line']:
                        any_line = [0]
                        any_line[0], = ax.plot(range(self.k['x_high']),
                                               self.k['iter_any'],
                                               c=self.k['anyc'])
                    if self.k['legend'] is not None:
                        self.lines[iii].set_label(self.k['legend'][iii])
                    if self.k['alpha'] is not None:
                        self.lines[iii].set_alpha(self.k['alpha'][iii])
                plt.axis([self.k['x_low'], self.k['x_high'], self.k['y_low'],
                          self.k['y_high']])
                plt.legend(fontsize=18)
                plt.tight_layout()
        grid(self.k['grid'])
        plt.pause(0.01)
        if not self.k['auto_xrange']:
            self.plot = self.plot2
            self.multi_plot = self.multi_plot2  # self.plot2 = self.plot

    def plot(self, list_data: Sequence):
        self.lines[0].set_xdata(range(len(list_data)))
        self.lines[0].set_ydata(
            list_data)  # self.lines[0].set_xlim([0, len(list_data)])

    def multi_plot(self, lists_data: Tuple[Sequence, ...]):
        for lists in range(len(lists_data)):
            # plt.xlim([0, len(lists_data[lists])])
            self.lines[lists].set_data(range(len(lists_data[lists])),
                                       lists_data[lists])

    def subplot(self, lists: Sequence):
        if len(lists) != self.n_line:
            raise AttributeError(
                f'len ("lists"): {len(lists)} != len ("self.n_line"): {self.n_line}')
        else:
            for iii in range(self.n_line):
                self.lines[iii].set_data(range(len(lists[iii])), lists[iii])

    def plot2(self, list_data: Sequence, x_list: Sequence):
        self.lines[0].set_data(x_list, list_data)

    def multi_plot2(self, lists_data: Tuple[Sequence, ...],
                    x_lists: Tuple[Sequence, ...]):
        if len(lists_data) != len(x_lists):
            raise AttributeError(
                f'len(lists_data): {len(lists_data)} != len(x_lists): {len(x_lists)}')
        for lists in range(len(lists_data)):
            self.lines[lists].set_data(x_lists[lists], lists_data[lists])

    # def scatter(self, xlist, ylist):
    #     self.lines[0].

    def save(self, nm_fig: str):
        self.fig.savefig(nm_fig)
        plt.pause(0.01)


if __name__ == "__main__":
    lg = Log(n_line=1, num_fig=0, figsize=(12, 8), y_low=-1.0, x_high=100,
             grid=True, c='green', inf=str(1))
    print(f'**kwargs: {lg.k}')
    # print(lg.plot())
    lg.plot(np.sin(np.arange(10)))
    plt.show()  # print(lg.__new__.__annotations__)
