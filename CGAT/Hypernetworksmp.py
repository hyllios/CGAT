# Uses code from https://github.com/vsitzmann/scene-representation-networks
'''Pytorch implementations of hyper-network modules.'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils

import numpy as np

import math
import numbers

import functools


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.net(input)


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(
            FCLayer(
                in_features=in_features,
                out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=hidden_ch))

        if outermost_linear:
            self.net.append(
                nn.Linear(
                    in_features=hidden_ch,
                    out_features=out_features))
        else:
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                a=0.0,
                nonlinearity='leaky_relu',
                mode='fan_in')

    def forward(self, input):
        return self.net(input)


class HyperLayer(nn.Module):
    '''A hypernetwork that predicts a single Dense Layer, including LayerNorm and a ReLU.'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch):
        super().__init__()

        self.hyper_linear = HyperLinear(
            in_ch=in_ch,
            out_ch=out_ch,
            hyper_in_ch=hyper_in_ch,
            hyper_num_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_ch=hyper_hidden_ch)
        self.norm_nl = nn.Sequential(
            nn.LayerNorm([out_ch], elementwise_affine=False),
            #            nn.ReLU(inplace=True)
            nn.Tanh()
        )

    def forward(self, hyper_input):
        '''
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        '''
        return nn.Sequential(self.hyper_linear(hyper_input), self.norm_nl)


class HyperFC(nn.Module):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''

    def __init__(self,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=False):
        super().__init__()

#        PreconfHyperLinear = partialclass(HyperLinear,
#                                          hyper_in_ch=hyper_in_ch,
#                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
#                                          hyper_hidden_ch=hyper_hidden_ch)
#        PreconfHyperLayer = partialclass(HyperLayer,
#                                          hyper_in_ch=hyper_in_ch,
#                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
#                                          hyper_hidden_ch=hyper_hidden_ch)

        self.layers = nn.ModuleList()
        self.layers.append(
            HyperLayer(
                in_ch=in_ch,
                out_ch=hidden_ch,
                hyper_in_ch=hyper_in_ch,
                hyper_num_hidden_layers=hyper_num_hidden_layers,
                hyper_hidden_ch=hyper_hidden_ch))

        for i in range(num_hidden_layers):
            self.layers.append(
                HyperLayer(
                    in_ch=hidden_ch,
                    out_ch=hidden_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))

        if outermost_linear:
            self.layers.append(
                HyperLinear(
                    in_ch=hidden_ch,
                    out_ch=out_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))
        else:
            self.layers.append(
                HyperLayer(
                    in_ch=hidden_ch,
                    out_ch=out_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))

    def forward(self, hyper_input):
        '''
        :param hyper_input: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        '''
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](hyper_input))

        return nn.Sequential(*net)


class BatchLinear(nn.Module):
    def __init__(self,
                 weights,
                 biases):
        '''Implements a batch linear layer.
        :param weights: Shape: (batch, out_ch, in_ch)
        :param biases: Shape: (batch, 1, out_ch)
        '''
        super().__init__()

        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return "BatchLinear(in_ch=%d, out_ch=%d)" % (
            self.weights.shape[-1], self.weights.shape[-2])

    def forward(self, input):
        output = input.matmul(self.weights.permute(
            *[i for i in range(len(self.weights.shape) - 2)], -1, -2))
        output += self.biases
        return output


def last_hyper_layer_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,
            nonlinearity='leaky_relu',
            mode='fan_in')
        m.weight.data *= 1e-1


class HyperLinear(nn.Module):
    '''A hypernetwork that predicts a single linear layer (weights & biases).'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch):

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.hypo_params = FCBlock(in_features=hyper_in_ch,
                                   hidden_ch=hyper_hidden_ch,
                                   num_hidden_layers=hyper_num_hidden_layers,
                                   out_features=(in_ch * out_ch) + out_ch,
                                   outermost_linear=True)
        self.hypo_params[-1].apply(last_hyper_layer_init)

    def forward(self, hyper_input):
        hypo_params = self.hypo_params(hyper_input.cuda())

        # Indices explicit to catch erros in shape of output layer
        weights = hypo_params[..., :self.in_ch * self.out_ch]
        biases = hypo_params[..., self.in_ch *
                             self.out_ch:(self.in_ch * self.out_ch) + self.out_ch]

        biases = biases.view(*(biases.size()[:-1]), 1, self.out_ch)
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)

        return BatchLinear(weights=weights, biases=biases)


class H_Net_0(nn.Module):
    def __init__(self, hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=True):
        super(H_Net_0, self).__init__()
        self.Hyper = HyperFC(hyper_in_ch,
                             hyper_num_hidden_layers,
                             hyper_hidden_ch,
                             hidden_ch,
                             num_hidden_layers,
                             in_ch,
                             out_ch,
                             outermost_linear=True)
        self.out_ch = out_ch

    def forward(self, h_0, x):
        NN = self.Hyper(h_0)
        return NN(
            x.view(
                x.shape[0],
                1,
                x.shape[1])).view(
            x.shape[0],
            self.out_ch)


class H_Net(nn.Module):
    def __init__(self, hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=True):
        super(H_Net, self).__init__()
        self.Hyper = HyperFC(hyper_in_ch,
                             hyper_num_hidden_layers,
                             hyper_hidden_ch,
                             hidden_ch,
                             num_hidden_layers,
                             in_ch,
                             out_ch,
                             outermost_linear=True)
        self.damping = nn.Parameter(torch.rand(1))
        self.out_ch = out_ch

    def forward(self, h_0, h_t, x):
        with torch.no_grad():
            self.damping.data = self.damping.data.clamp(0.0, 1.0)
        NN = self.Hyper(self.damping * h_0 + (1 - self.damping) * x)
        return NN(x.view(x.shape[0], 1, x.shape[1])).view(x.shape[0], self.out_ch)
