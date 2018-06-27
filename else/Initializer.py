import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def uniform(low, high, size: Tuple[int]):
    return np.random.uniform(low, high, size)


def normal(loc, scale, size: Tuple[int]):
    return np.random.normal(loc, scale, size)
