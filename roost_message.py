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

import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_max, scatter_add, \
    scatter_mean
import json



class Featuriser(object):
    """
    Base class for featurising nodes and edges.
    """

    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, "{} is not an allowed atom type".format(
            key)
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])


class LoadFeaturiser(Featuriser):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Notes
    ---------
    For the specific composition net application the keys are concatenated
    strings of the form "NaCl" where the order of concatenation matters.
    This is done because the bond "ClNa" has the opposite dipole to "NaCl"
    so for a general representation we need to be able to asign different
    bond features for different directions on the multigraph.

    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super(LoadFeaturiser, self).__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)



class MessageLayer(nn.Module):
    """
    Class defining the message passing operation on the composition graph
    """

    def __init__(self, fea_len, num_heads=1):
        """
        Inputs
        ----------
        fea_len: int
            Number of elem hidden features.
        """
        super(MessageLayer, self).__init__()

        # Pooling and Output
        hidden_ele = [256]
        hidden_msg = [256]
        self.pooling = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(2 * fea_len, 1, hidden_ele),
            message_nn=SimpleNetwork(2 * fea_len, fea_len, hidden_msg),
            # message_nn=nn.Linear(2*fea_len, fea_len),
            # message_nn=nn.Identity(),
        ) for _ in range(num_heads)])

    def forward(self, elem_weights, elem_in_fea,
                self_fea_idx, nbr_fea_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elems (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each elem
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each elem
        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Atom hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)
        #print('fea shape after cat',fea.shape)
        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(fea=fea,
                                     index=self_fea_idx,
                                     weights=elem_nbr_weights))

        # # Concatenate
        # fea = torch.cat(head_fea, dim=1)
        fea = torch.mean(torch.stack(head_fea), dim=0)
        #print(fea.shape, elem_in_fea.shape)
        return fea + elem_in_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Roost(nn.Module):
    """
    Create a neural network for predicting total material properties.
    The Roost model is comprised of a fully connected network
    and message passing graph layers.
    The message passing layers are used to determine a descriptor set
    for the fully connected network. Critically the graphs are used to
    represent (crystalline) materials in a structure agnostic manner
    but contain trainable parameters unlike other structure agnostic
    approaches.
    """

    def __init__(self, orig_elem_fea_len, elem_fea_len, n_graph):
        """
        Initialize CompositionNet.
        Parameters
        ----------
        n_h: Number of hidden layers after pooling
        Inputs
        ----------
        orig_elem_fea_len: int
            Number of elem features in the input.
        elem_fea_len: int
            Number of hidden elem features in the graph layers
        n_graph: int
            Number of graph layers
        """
        super(Roost, self).__init__()

        # apply linear transform to the input to get a trainable embedding
        self.embedding = nn.Linear(orig_elem_fea_len, elem_fea_len - 1)

        # create a list of Message passing layers

        msg_heads = 1
        self.graphs = nn.ModuleList(
            [MessageLayer(elem_fea_len, msg_heads)
             for i in range(n_graph)])

        # define a global pooling function for materials
        mat_heads = 1
        mat_hidden = [256]
        # msg_hidden = [256]
        self.cry_pool = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(elem_fea_len, 1, mat_hidden),
            # message_nn=SimpleNetwork(elem_fea_len, 20, msg_hidden),
            # message_nn=nn.Linear(elem_fea_len, elem_fea_len),
            message_nn=nn.Identity(),
        ) for _ in range(mat_heads)])

        # define an output neural network
        # out_hidden = [512, 256, 128, 64]

    def forward(self, elem_weights, orig_elem_fea, self_fea_idx,
                nbr_fea_idx, crystal_elem_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elems (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        orig_elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the elem each of the M bonds correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of of the neighbours of the M bonds connect to
        elem_bond_idx: list of torch.LongTensor of length C
            Mapping from the bond idx to elem idx
        crystal_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx
        Returns
        -------
        out: nn.Variable shape (C,)
            Atom hidden features after message passing
        """

        # embed the original features into the graph layer description
        elem_fea = self.embedding(orig_elem_fea)

        # do this so that we can examine the embeddings without
        # influence of the weights
        #print(elem_fea.shape, elem_weights.shape)
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the graph message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea,
                                  self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(attnhead(fea=elem_fea,
                                     index=crystal_elem_idx,
                                     weights=elem_weights))

        crys_fea = torch.mean(torch.stack(head_fea), dim=0)
        # crys_fea = torch.cat(head_fea, dim=1)

        # apply neural network to map from learned features to target

        return crys_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedMeanPooling(torch.nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super(WeightedMeanPooling, self).__init__()

    def forward(self, fea, index, weights):
        fea = weights * fea
        return scatter_mean(fea, index, dim=0)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedAttention(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn, num_heads=1):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(WeightedAttention, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, fea, index, weights):
        """ forward pass """

        gate = self.gate_nn(fea)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-13)

        fea = self.message_nn(fea)
        # print(fea.shape)
        out = scatter_add(gate * fea, index, dim=0)
        # print(out.shape)
        return out

    def __repr__(self):
        return '{}(gate_nn={})'.format(self.__class__.__name__,
                                       self.gate_nn)


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

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU()
                                   for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
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

    def forward(self, fea):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom"s M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    cry_base_idx = 0
    for i, (atom_weights, atom_fea, self_fea_idx,
            nbr_fea_idx) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        cry_base_idx += n_i
#        print('hal',torch.cat(batch_atom_weights, dim=0).shape,torch.cat(batch_atom_fea, dim=0).shape )
    return (torch.cat(batch_atom_weights, dim=0).view(-1, 1),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx))
