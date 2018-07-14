# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import function as OF


class ESN(nn.Module):

    # Constructor
    def __init__(self, input_dim, output_dim, feedback_dim, w=None, w_in=None, w_fb=None,
                 leak_rate=None, w_sparsity=None, w_in_sparsity=None, w_fb_sparsity=None):
        super().__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedback_dim = feedback_dim
        self.w_sparsity = w_sparsity
        self.w_in_sparsity = w_sparsity
        self.w_fb_sparsity = w_sparsity
        self.leak_rate = leak_rate

        # Define connect
        self.fc_in = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.fc_fb = nn.Linear(self.feedback_dim, self.output_dim, bias=False)
        self.fc_rec = nn.Linear(self.output_dim, self.output_dim, bias=False)

        # Initialize W matrix
        # todo apply()？を使ってLinearの時点でこの関数で重み値を初期化したほうが二度手間でない
        # todo スパース結合を作ってくれるtorch.nn.Embeddingクラスがあるらしい
        self.fc_in.weight.data = self.generate_w(w_in, self.w_in_sparsity,
                                                 self.output_dim, self.input_dim)
        self.fc_fb.weight.data = self.generate_w(w_fb, self.w_fb_sparsity,
                                                 self.output_dim, self.feedback_dim)
        self.fc_rec.weight.data = self.generate_w(w, self.w_sparsity,
                                                  self.output_dim, self.output_dim)

        print(f'w: {self.fc_rec.weight.data}')

    # end __init__

    ##################################
    # PROPERTIES
    ##################################

    ##################################
    # PUBLIC
    ##################################

    # Forward
    def forward(self, u, x, old_y, fb):
        self.u = self.leak_rate * self.u + \
                 (1 - self.leak_rate) * (self.fc_in(x) + self.fc_rec(old_y) + self.fc_fb(fb))
        y = F.tanh(u)
        return y

    #####################################
    # STATIC
    #####################################

    # Generate W matrix
    @staticmethod
    def generate_w(w, sparsity, *size):
        if callable(w):
            if sparsity is None:
                return w(size)
            else:
                s_w = np.random.choice([0.0, 1.0], size, p=[1.0 - sparsity, sparsity])
                s_w = torch.FloatTensor(s_w)
                s_w[s_w == 1] = w((len(s_w[s_w == 1]),))
        else:
            return torch.zeros(size)


    def init_hidden(self):

        return torch.zeros(self.outptu_dim)


if __name__ == '__main__':
    esn = ESN(2, 5, 3, w=OF.generate_uw(-3.0, 3.0), w_sparsity=0.1)
    y = esn.forward()
