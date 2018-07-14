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
    def __init__(self, input_dim, output_dim, feedback_dim, w=None, leak_rate=None,
                 w_sparsity=None):
        super().__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedback_dim = feedback_dim
        self.leak_rate = leak_rate

        # Define connection
        self.fc_in = nn.Linear(self.input_dim, self.output_dim) # fc : Full connection
        self.fc_fb = nn.Linear(self.feedback_dim, self.output_dim)
        self.fc_rec = nn.Linear(self.output_dim, self.output_dim)

        # Initialize W matrix
        if w_sparsity is None:
            if callable(w):
                self.fc_rec.weight.data = w((self.output_dim, self.output_dim))
            else:
                self.fc_rec.weight.data.uniform_(-1.0, 1.0)

        else:
            self.w = np.random.choice([0.0, 1.0], (self.output_dim, self.output_dim),
                                      p=[1.0 - w_sparsity, w_sparsity])
            self.w = torch.FloatTensor(self.w)

            if callable(w):
                self.w[self.w == 1] = w((len(self.w[w == 1]), ))
            else:
                self.w[self.w == 1].uniform_(-1.0, 1.0, (len(self.w[w == 1]), ))

        print(f'w: {self.fc_rec.weight.data}')

    # end __init__

    ##################################
    # PROPERTIES
    ##################################

    ##################################
    # PUBLIC
    ##################################

    def forward(self, u, x, old_y, fb):
        self.u = self.leak_rate * self.u + (1 - self.leak_rate) * (
                    self.fc_in(x) + self.fc_rec(old_y) + self.fc_fb(fb))
        y = F.tanh(u)
        return y

    def init_hidden(self):

        return torch.zeros(self.outptu_dim)




if __name__ == '__main__':
    esn = ESN(2, 5, 3, w=OF.generate_uw(-3.0, 3.0), w_sparsity=0.1)
    y = esn.forward()
