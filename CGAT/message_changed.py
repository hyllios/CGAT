import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_max, scatter_add, \
    scatter_mean

"""
MIT License
Copyright (c) 2019-2020 Rhys Goodall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(SimpleNetwork, self).__init__()

        dims = [input_dim] + hidden_layer_dims
        # print(dims, output_dim)
        # print(dims)

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU()
                                   for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            # print('fea',fea.shape)
            fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Rezero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.alpha * x

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layer_dims,
            if_rezero=False):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
        #                           for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                      if (dims[i] != dims[i + 1])
                                      else nn.Identity()
                                      for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)
        self.if_rezero = if_rezero
        if (self.if_rezero):
            self.rezeros = nn.ModuleList(
                [Rezero() for _ in range(len(dims) - 1)])

    def forward(self, fea, *, last_layer=True):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        if (not self.if_rezero):
            for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
                fea = act(fc(fea)) + res_fc(fea)
        else:
            for fc, res_fc, act, rez in zip(
                    self.fcs, self.res_fcs, self.acts, self.rezeros):
                fea = rez(act(fc(fea))) + res_fc(fea)

        if last_layer:
            return self.fc_out(fea)
        else:
            return fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
