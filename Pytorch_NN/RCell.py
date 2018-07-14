# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from function import n_rand, u_rand


# Recurrent neural network layer
class RCell(nn.Module):
    """
    Recurrent Neural Network layer
    """

    # Constructor
    def __init__(self, input_dim, output_dim, w=None, w_sparsity=False):

        super().__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_sparsity = w_sparsity
        self._w = self.generate_w(self.output_dim, w, self.w_sparsity)
        self.l_r = nn.Linear(self.input_dim, self.output_dim)

    # end __init__

    #######################################
    # PROPERTIES
    #######################################

    @property
    def w(self):
        return self._w

    #######################################
    # SETTERS
    #######################################

    @w.setter
    def w(self, w):
        self._w = w

    #######################################
    # PUBLIC
    #######################################

    # Forward
    def forward(self, input):
        pass

    ######################################
    # STATIC
    ######################################

    # Generate W matrix
    @staticmethod
    def generate_w(output_dim, w=None, w_sparsity=None):
        if w is None:
            w = u_rand((output_dim, output_dim), low=-1.0, high=1.0)
        else:
            if w_sparsity is not None:
                ch = np.random.choice([0.0, 1.0], (output_dim, output_dim),
                                      p=[1.0 - w_sparsity, w_sparsity])
                w = w * ch
                w = torch.from_numpy(w.astype(np.float32))

        return w


if __name__ == "__main__":
    pass
