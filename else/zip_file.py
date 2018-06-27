import numpy as np
import matplotlib.pyplot as plt
import zipfile


def w(zipname: str, filename: str):
    with zipfile.ZipFile(f'{zipname}.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(filename)


def a(zipname: str, filename: str):
    with zipfile.ZipFile(f'{zipname}.zip', 'a', zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(filename)


if __name__ == '__main__':
    # w('test', 'p_run.py')
    a('test3', 'Reservoir_ReNN.py')
