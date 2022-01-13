from torch_geometric.data import Data
import gzip as gz
import os
import sys

import functools
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from roost_message import LoadFeaturiser


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset 
    """

    def __init__(self, data, fea_path, radius=8.0, max_neighbor_number=12, target='e_above_hull'):
        """
                Constructs dataset
        Args:
            data: expects either a gzipped pickle of the dictionary or a dictionary
                  with the keys 'batch_comp', 'comps', 'target', 'input'
            fea_path:
                  path  to file containing the element embedding information
            radius:
                  cutoff radius
            max_neighbor_number:
                  maximum number of neighbors used during message passing
            target:
                  name of training/validation/testing target
        Returns:
        """

        if isinstance(data, str):
            assert os.path.exists(data), \
                "{} does not exist!".format(data)
            self.data = pickle.load(gz.open(data, "rb"))
        else:
            self.data = data

        self.radius = radius
        self.max_num_nbr = max_neighbor_number
        if(self.data['input'].shape[0] > 3):
            self.format = 1
        else:
            self.format = 0
        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.atom_features = LoadFeaturiser(fea_path)
        self.atom_fea_dim = self.atom_features.embedding_size
        self.target = target

    def __len__(self):
        """Returns length of dataset"""
        return len(self.data['target'][self.target])

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        composition = self.data['batch_comp'][idx]
        elements = self.data['comps'][idx]
        try:
            elements = elements.tolist()
        except BaseException:
            pass
        if(isinstance(elements[0], list) or isinstance(elements[0], tuple)):
            elements = [el[0] for el in elements]
        N = len(elements)
        comp = {}
        weights = []
        elements2 = []
        for el in elements:
            comp[el] = elements.count(el)

        for k, v in comp.items():
            weights.append(v / len(elements))
            elements2.append(k)
        env_idx = list(range(len(elements2)))
        self_fea_idx_c = []
        nbr_fea_idx_c = []
        nbrs = len(elements2) - 1
        for i, _ in enumerate(elements2):
            self_fea_idx_c += [i] * nbrs
            nbr_fea_idx_c += env_idx[:i] + env_idx[i + 1:]

        atom_fea_c = np.vstack([self.atom_features.get_fea(element)
                                for element in elements2])
        atom_weights_c = torch.Tensor(weights)
        atom_fea_c = torch.Tensor(atom_fea_c)
        self_fea_idx_c = torch.LongTensor(self_fea_idx_c)
        nbr_fea_idx_c = torch.LongTensor(nbr_fea_idx_c)

        if(self.format == 0):
            try:
                atom_fea = np.vstack([self.atom_features.get_fea(element)
                                      for element in elements])
            except AssertionError:
                print(composition)
                sys.exit()

            target = self.data['target'][self.target][idx]
            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.LongTensor(
                self.data['input'][0][idx][:, 0:self.max_num_nbr].flatten().astype(int))
            nbr_fea_idx = torch.LongTensor(
                self.data['input'][2][idx][:, 0:self.max_num_nbr].flatten().astype(int))
            self_fea_idx = torch.LongTensor(
                self.data['input'][1][idx][:, 0:self.max_num_nbr].flatten().astype(int))
            target = torch.Tensor([target])
        else:
            try:
                atom_fea = np.vstack([self.atom_features.get_fea(
                    elements[i]) for i in range(len(elements))])
            except AssertionError:
                print(composition)
                sys.exit()

            target = self.data['target'][self.target][idx]
            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.LongTensor(
                self.data['input'][idx][0][:, 0:self.max_num_nbr].flatten())
            nbr_fea_idx = torch.LongTensor(
                self.data['input'][idx][2][:, 0:self.max_num_nbr].flatten())
            self_fea_idx = torch.LongTensor(
                self.data['input'][idx][1][:, 0:self.max_num_nbr].flatten())
            target = torch.Tensor([target])
        if self.target!='volume':
            return Data(x=atom_fea, edge_index=torch.stack((self_fea_idx, nbr_fea_idx)), edge_attr=nbr_fea,
                    y=target * N), (atom_weights_c, atom_fea_c, self_fea_idx_c, nbr_fea_idx_c)
        else:
            return Data(x=atom_fea, edge_index=torch.stack((self_fea_idx, nbr_fea_idx)), edge_attr=nbr_fea,
                y=target), (atom_weights_c, atom_fea_c, self_fea_idx_c, nbr_fea_idx_c)
