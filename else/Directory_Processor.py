# coding=utf-8

import os
from datetime import datetime


class DirectoryProcessing:
    _root_name = 'data'

    def __init__(self):
        self.now = datetime.now()
        self.subroot_name = '{0:%m%d_%H%M_%S}'.format(self.now)
        self.subroot_path = f'{self.root_name}/{self.subroot_name}/'
        self.data_folders = {'root': self._root_name, 'subroot': self.subroot_path}
        os.makedirs(self.subroot_path, exist_ok=True)

    @property
    def root_name(self):
        return self._root_name

    @root_name.setter
    def root_name(self, name):
        self._root_name = name + str('{0: %Y}'.format(self.now))

    # def setting_directory(self):
    #     os.makedirs(self.subroot_path, exist_ok=True)

    def adding_datafolder(self, *folder_name):
        for name in folder_name:
            self.data_folders[name] = f'{self.subroot_path}/{name}/'
            os.makedirs(f'{self.subroot_path}/{name}/', exist_ok=True)


if __name__ == "__main__":
    dirs = DirectoryProcessing()
    dirs.adding_datafolder('test', 'test2')
