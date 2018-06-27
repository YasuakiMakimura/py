# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class OptionSource:
    _color = None
    _filename = None
    _title = None

    @property
    def line_color(self):
        return self._color

    @property
    def file_name(self):
        return self._filename

    @property
    def title(self):
        return self._title

    @line_color.setter
    def line_color(self, color_name: str):
        self._color = color_name

    @file_name.setter
    def file_name(self, filename: str):
        self._filename = filename

    @title.setter
    def title(self, titlename: str):
        self._title = titlename


if __name__ == "__main__":
    pass
