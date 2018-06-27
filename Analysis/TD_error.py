import numpy as np
import contextlib
from bokeh.plotting import figure
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.io import export_png
from collections import deque
from Bokeh_Option import OptionSource
from typing import ContextManager


# class Option:
#     def __init__(self):
#         self.option_source = OptionSource()
#
#     @property
#     def filename(self):
#         return self.option_source.title
#
#     @filename.setter
#     def filename(self, filename):
#         self.option_source.file_name = filename
#

class Plot:
    def __init__(self):
        self.hanger = deque([])
        self.figure = figure(plot_width=400, plot_height=400)

    def store(self, td_value):
        self.hanger.append(td_value)

    def plot(self, filename: str, save_with_png: bool=False):
        def export_to_png():
            export_png(self.figure, filename=f'{filename}.png')

        def show_with_html():
            return show(self.figure)

        self.figure.line(range(len(self.hanger)), self.hanger)
        if save_with_png:
            return export_to_png()
        else:
            return show_with_html()


if __name__ == '__main__':
    # todo まずwith文の中でPlot classをインスタンス、そしてその中のインスタンス変数のoptionにグラフの各オプションをセットする。そして、後処理にsetup関数を実行して各オプションをインスタンス変数として追加する。
    import random

    boke = Plot()
    for i in range(20):
        print(i)
        rnd = random.uniform(-1, 1)
        boke.store(rnd)
    # boke.plot(pngsave=True)
    boke.plot('test', save_with_png=False)
