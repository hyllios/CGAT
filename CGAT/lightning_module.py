"""
Example template for defining a system
"""
from roost_message import collate_batch
import importlib

from lambs import JITLamb
import numpy as np

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse

from sklearn.model_selection import train_test_split as split

from utils import RobustL1, RobustL2, cyclical_lr
from torch_geometric.data import Batch
from data import CompositionData

from pytorch_lightning.core import LightningModule


def collate_fn(datalist):
    return datalist


class LightningModel(LightningModule):
    """
    Lightning model for CGAtNet defined in hparams.version
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super().__init__()
        self.hparams = hparams
        if self.hparams.train:
            dataset = CompositionData(
                data=self.hparams.data_path,
                fea_path=self.hparams.fea_path,
                max_neighbor_number=self.hparams.max_nbr)
            indices = list(range(len(dataset)))
            train_idx, test_idx = split(indices, random_state=self.hparams.seed,
                                        test_size=self.hparams.test_size)
            train_set = torch.utils.data.Subset(dataset, train_idx)
            self.test_set = torch.utils.data.Subset(dataset, test_idx)
            indices = list(range(len(train_set)))
            train_idx, val_idx = split(indices, random_state=self.hparams.seed,
                                       test_size=self.hparams.val_size / (1 - self.hparams.test_size))
            train_set_2 = torch.utils.data.Subset(train_set, train_idx)
            self.val_subset = torch.utils.data.Subset(train_set, val_idx)

            # Use train_percentage to get errors for different training set sizes
            # but same test and validation sets
            if self.hparams.train_percentage != 0.0:
                indices = list(range(len(train_set_2)))
                train_idx, rest_idx = split(
                    indices, random_state=self.hparams.seed, test_size=1.0 - self.hparams.train_percentage / (
                        1 - self.hparams.val_size - self.hparams.test_size))
                self.train_subset = torch.utils.data.Subset(train_set_2, train_idx)
            else:
                self.train_subset = train_set_2

            print('Normalization started')
            def collate_fn2(data_list): return [el[0].y for el in data_list]
            sample_target = torch.cat(collate_fn2(self.train_subset))
            self.mean = torch.mean(sample_target, dim=0, keepdim=False)
            self.std = torch.std(sample_target, dim=0, keepdim=False)
            print('mean:', self.mean, 'std:', self.std)
            print('normalization ended')

        # select loss function
        if not self.hparams.std_loss:
            print('robust loss')
            if self.hparams.loss == "L1":
                self.criterion = RobustL1
            elif self.hparams.loss == "L2":
                self.criterion = RobustL2
        elif self.hparams.std_loss:
            print('No robust loss function')
            if self.hparams.loss == 'L1':
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()
        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def norm(self, tensor):
        return (tensor - self.mean.cuda()) / self.std.cuda()

    def denorm(self, normed_tensor):
        return normed_tensor * self.std.cuda() + self.mean.cuda()

    def __build_model(self):
        """
        Layout model
        :return:
        """
        gat = importlib.import_module(self.hparams.version)
        self.model = gat.CGAtNet(200,
                                 elem_fea_len=self.hparams.atom_fea_len,
                                 n_graph=self.hparams.n_graph,
                                 rezero=self.hparams.rezero,
                                 mean_pooling=not self.hparams.mean_pooling,
                                 neighbor_number=self.hparams.max_nbr,
                                 msg_heads=self.hparams.msg_heads,
                                 update_edges=self.hparams.update_edges,
                                 vector_attention=self.hparams.vector_attention,
                                 global_vector_attention=self.hparams.global_vector_attention,
                                 n_graph_roost=self.hparams.n_graph_roost)  # , self.hparams.dropout)

        params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('this model has {0:1d} parameters '.format(params))
    # ---------------------
    # TRAINING
    # ---------------------

    def evaluate(self, batch):
        device = next(self.model.parameters()).device
        b_comp, batch = [el[1] for el in batch], [el[0] for el in batch]
        batch = (Batch.from_data_list(batch)).to(device)
        b_comp = collate_batch(b_comp)
        b_comp = (tensor.to(device) for tensor in b_comp)
        output, log_std = self.model(batch, b_comp).chunk(2, dim=1)
        target = batch.y.view(len(batch.y), 1)
        target_norm = self.norm(target)
        pred = self.denorm(output.data)
        return output, log_std, pred, target, target_norm

    def forward(self, batch, batch_idx):
        """
        Use for prediction with a dataloader
        """
        _, log_std, pred, _, _ = self.evaluate(batch)
        return pred

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        output, log_std, pred, target, target_norm = self.evaluate(batch)

        # calculate loss
        if not self.hparams.std_loss:
            loss = self.criterion(output, log_std, target_norm)
        else:
            loss = self.criterion(output, target_norm)

        mae_error = mae(pred, target)
        rmse_error = mse(pred, target).sqrt_()
        self.log('train_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        self.log('train_mae',
                 mae_error,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        self.log('train_rmse',
                 rmse_error,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        return loss

    def validation_step(self, batch):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        output, log_std, pred, target, target_norm = self.evaluate(batch)

        if not self.hparams.std_loss:
            val_loss = self.criterion(output, log_std, target_norm)
        else:
            val_loss = self.criterion(output, target_norm)

        val_mae = mae(pred, target)
        val_rmse = mse(pred, target).sqrt_()
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)
        self.log('val_rmse', val_rmse, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        output, log_std, pred, target, target_norm = self.evaluate(batch)

        if not self.hparams.std_loss:
            test_loss = self.criterion(output, log_std, target_norm)
        else:
            test_loss = self.criterion(output, target_norm)

        test_mae = mae(pred, target)
        test_rmse = mse(pred, target).sqrt_()
        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True)
        self.log('test_mae', test_mae, on_epoch=True, sync_dist=True)
        self.log('test_rmse', test_rmse, on_epoch=True, sync_dist=True)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        # Select Optimiser
        if self.hparams.optim == "SGD":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.hparams.learning_rate,
                                  weight_decay=self.hparams.weight_decay,
                                  momentum=self.hparams.momentum)
        elif self.hparams.optim == "Adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.hparams.learning_rate,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim == "LAMB":
            optimizer = JITLamb(self.model.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        else:
            raise NameError(
                "Only SGD, Adam, AdamW, Lambs are allowed as --optim")

        if self.hparams.clr:
            clr = cyclical_lr(period=self.hparams.clr_period,
                              cycle_mul=0.1,
                              tune_mul=0.05,)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=0.1,
                                                                   patience=5,
                                                                   verbose=False,
                                                                   threshold=0.0002,
                                                                   threshold_mode='rel',
                                                                   cooldown=0,
                                                                   eps=1e-08)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        params = {"batch_size": self.hparams.batch_size,
                  "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "shuffle": True,
                  "drop_last": True
                  }
        print('length of train_subset', len(self.train_subset))
        train_generator = DataLoader(
            self.train_subset, collate_fn=collate_fn, **params)
        return train_generator

    def val_dataloader(self):
        params = {"batch_size": self.hparams.batch_size,
                  #              "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "drop_last": True,
                  "shuffle": False}
        val_generator = DataLoader(
            self.val_subset,
            collate_fn=collate_fn,
            **params)
        print('length of val_subset', len(self.val_subset))
        return val_generator

    def test_dataloader(self):
        params = {"batch_size": self.hparams.batch_size,
                  #              "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "drop_last": True,
                  "shuffle": False}
        test_generator = DataLoader(
            self.test_set, collate_fn=collate_fn, **params)
        print('length of test_subset', len(self.test_set))
        return test_generator

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument("--data-path",
                            type=str,
                            default="data.pickle.gz",
                            metavar="PATH",
                            help="dataset path")
        parser.add_argument("--fea-path",
                            type=str,
                            default="data/embeddings/onehot-embedding.json",
                            metavar="PATH",
                            help="atom feature path")
        parser.add_argument("--version",
                            type=str,
                            default="CGAT",
                            help="module from which to load CGAtNet class")
        parser.add_argument("--nbr-embedding-size",
                            default=512,
                            type=int,
                            help="size of edge embedding")
        parser.add_argument("--msg-heads",
                            default=5,
                            type=int,
                            help="number of attention-heads in message passing/final pooling layer")
        parser.add_argument("--workers",
                            default=0,
                            type=int,
                            metavar="N",
                            help="number of data loading workers (default: 0), crashes on some machines if used")
        parser.add_argument("--batch-size",
                            default=64,
                            type=int,
                            metavar="N",
                            help="mini-batch size (default: 128), when using multiple gpus the actual batch-size"
                                 " is --batch-size*n_gpus")
        parser.add_argument("--val-size",
                            default=0.1,
                            type=float,
                            metavar="N",
                            help="proportion of data used for validation")
        parser.add_argument("--test-size",
                            default=0.1,
                            type=float,
                            metavar="N",
                            help="proportion of data for testing")
        parser.add_argument("--max-nbr",
                            default=24,
                            type=int,
                            metavar="max_N",
                            help="num of neighbors maximum depends on the number set during the feature calculation")
        parser.add_argument("--epochs",
                            default=390,
                            type=int,
                            metavar="N",
                            help="number of total epochs to run")
        parser.add_argument("--loss",
                            default="L1",
                            type=str,
                            metavar="str",
                            help="choose a (Robust if std-loss False) Loss Function; L2 or L1")
        parser.add_argument("--optim",
                            default="AdamW",
                            type=str,
                            metavar="str",
                            help="choose an optimizer; SGD, Adam or AdamW")
        parser.add_argument("--learning-rate", "--lr",
                            default=0.000125,
                            type=float,
                            metavar="float",
                            help="initial learning rate (default: 3e-4)")
        parser.add_argument("--momentum",
                            default=0.9,
                            type=float,
                            metavar="float [0,1]",
                            help="momentum (default: 0.9)")
        parser.add_argument("--weight-decay",
                            default=1e-6,
                            type=float,
                            metavar="float [0,1]",
                            help="weight decay (default: 0)")
        parser.add_argument("--atom-fea-len",
                            default=128,
                            type=int,
                            metavar="N",
                            help="size of node embedding")
        parser.add_argument("--n-graph",
                            default=5,
                            type=int,
                            metavar="N",
                            help="number of graph layers in CGAT model")
        parser.add_argument("--n-graph-roost",
                            default=3,
                            type=int,
                            metavar="N",
                            help="number of graph layers in roost module")
        parser.add_argument("--global_vector_attention",
                            action="store_false",
                            help="whether vector attention or scalar attention is used")
        parser.add_argument("--update_edges",
                            action="store_false",
                            help="whether edges are updated")
        parser.add_argument("--vector_attention",
                            action="store_false",
                            help="whether vector attention or scalar attention is used")
        parser.add_argument("--clr",
                            action="store_false",
                            help="use a cyclical learning rate schedule")
        parser.add_argument("--rezero",
                            action="store_false",
                            help="start residual layers with 0 as prefactor")
        parser.add_argument("--mean-pooling",
                            action="store_false",
                            help="chooses pooling variant")
        parser.add_argument("--std-loss",
                            action="store_false",
                            help="whether to choose a loss function that considers uncertainty")
        parser.add_argument("--clr-period",
                            default=130,
                            type=int,
                            help="how many epochs per learning rate cycle to perform")
        parser.add_argument("--train-percentage",
                            default=0.0,
                            type=float,
                            help="Percentage of the training data that is used for training (only use to get a training set size vs test error curve")
        parser.add_argument("--seed",
                            default=0,
                            type=int,
                            metavar="N",
                            help="seed for random number generator")
        parser.add_argument("--smoke-test",
                            action="store_true",
                            help="Finish quickly for testing")
        parser.add_argument("--train",
                            action="store_true",
                            help="if set to False datasets will not be loaded to speed up loading of the model")
        return parser
