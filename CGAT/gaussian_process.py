import gpytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse

from .lightning_module import LightningModel, collate_fn, collate_batch
from pytorch_lightning.core import LightningModule

from torch_geometric.data import Batch

from .data import CompositionData
from .utils import cyclical_lr

import glob
import os
from argparse import ArgumentParser
import datetime
import itertools

from sklearn.model_selection import train_test_split as split

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        # init base class
        distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, distribution)
        super(GPModel, self).__init__(strategy)

        # init mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # init likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, batch):
        # calculate means and covariances of the batch
        mean = self.mean_module(batch)
        covar = self.covar_module(batch)
        # return distribution
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def parameters(self, recurse: bool = True):
        return itertools.chain(super().parameters(recurse),
                               self.likelihood.parameters(recurse))


class GLightningModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # initialization of mean and standard deviation of the target data (needed for reloading without recalculation)
        self.mean = nn.parameter.Parameter(torch.zeros(1), requires_grad=False)
        self.std = nn.parameter.Parameter(torch.zeros(1), requires_grad=False)

        self.save_hyperparameters(hparams)

        # loading of cgat model (needed for calculating the embeddings)
        self.cgat_model = LightningModel.load_from_checkpoint(self.hparams.cgat_model, train=False)
        self.cgat_model.eval()
        self.cgat_model.cuda()

        # prepare tensor for inducing points
        embedding_dim = self.cgat_model.hparams.atom_fea_len * self.cgat_model.hparams.msg_heads
        self.inducing_points = nn.parameter.Parameter(torch.zeros((self.hparams.inducing_points, embedding_dim)),
                                                      requires_grad=False)

        # datasets are loaded for training or testing not needed in production
        if self.hparams.train:
            datasets = []
            # used for single file
            try:
                dataset = CompositionData(
                    data=self.hparams.data_path,
                    fea_path=self.hparams.fea_path,
                    max_neighbor_number=self.hparams.max_nbr,
                    target=self.hparams.target)
                print(self.hparams.data_path + ' loaded')
            # used for folder of dataset files
            except AssertionError:
                f_n = sorted([file for file in glob.glob(os.path.join(self.hparams.data_path, "*.pickle.gz"))])
                print("{} files to load".format(len(f_n)))
                for file in f_n:
                    try:
                        datasets.append(CompositionData(
                            data=file,
                            fea_path=self.hparams.fea_path,
                            max_neighbor_number=self.hparams.max_nbr,
                            target=self.hparams.target))
                        print(file + ' loaded')
                    except AssertionError:
                        print(file + ' could not be loaded')
                print("{} files succesfully loaded".format(len(datasets)))
                dataset = torch.utils.data.ConcatDataset(datasets)

            if self.hparams.test_path is None or self.hparams.val_path is None:
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
            else:
                test_data = torch.utils.data.ConcatDataset([CompositionData(data=file,
                                                                            fea_path=self.hparams.fea_path,
                                                                            max_neighbor_number=self.hparams.max_nbr,
                                                                            target=self.hparams.target)
                                                            for file in glob.glob(
                        os.path.join(self.hparams.test_path, "*.pickle.gz"))])
                val_data = torch.utils.data.ConcatDataset([CompositionData(data=file,
                                                                           fea_path=self.hparams.fea_path,
                                                                           max_neighbor_number=self.hparams.max_nbr,
                                                                           target=self.hparams.target)
                                                           for file in glob.glob(
                        os.path.join(self.hparams.val_path, "*.pickle.gz"))])

                train_set = dataset
                self.test_set = test_data
                train_set_2 = train_set
                self.val_subset = val_data

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
            self.hparams.train_size = len(self.train_subset)
            print('Normalization started')

            def collate_fn2(data_list):
                return [el[0].y for el in data_list]

            sample_target = torch.cat(collate_fn2(self.train_subset))
            self.mean = nn.parameter.Parameter(torch.mean(sample_target, dim=0, keepdim=False),
                                               requires_grad=False)
            self.std = nn.parameter.Parameter(torch.std(sample_target, dim=0, keepdim=False), requires_grad=False)
            print('mean: ', self.mean.item(), 'std: ', self.std.item())
            print('normalization ended')

            print('Calculating embedding of inducing points')
            # getting inducing_points
            loader = DataLoader(self.train_subset, batch_size=self.hparams.inducing_points, shuffle=True,
                                collate_fn=collate_fn)
            batch = next(iter(loader))
            del loader
            with torch.no_grad():
                self.inducing_points = nn.parameter.Parameter(
                    self.cgat_model.evaluate(batch, return_graph_embedding=True), requires_grad=False)
            print('Done')

        self.model = GPModel(self.inducing_points)
        # only init loss function for training
        self.criterion = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model, self.hparams.train_size)

    def norm(self, tensor):
        """
        normalizes tensor
        """
        return (tensor - self.mean.cuda()) / self.std.cuda()

    def denorm(self, normed_tensor):
        """
        return normalized tensor to original form
        """
        return normed_tensor * self.std.cuda() + self.mean.cuda()

    def evaluate(self, batch):
        with torch.no_grad():
            embeddings = self.cgat_model.evaluate(batch, return_graph_embedding=True)
        output = self.model(embeddings)
        pred = self.denorm(output.mean)
        lower, upper = output.confidence_region()

        device = next(self.model.parameters()).device
        b_comp, batch = [el[1] for el in batch], [el[0] for el in batch]
        batch = (Batch.from_data_list(batch)).to(device)

        target = batch.y.view(len(batch.y), 1)
        target_norm = self.norm(target)

        return output, (self.denorm(lower), self.denorm(upper)), pred, target, target_norm

    def forward(self, batch):
        _, _, pred, _, _ = self.evaluate(batch)
        return pred

    def training_step(self, batch, batch_idx):
        output, _, pred, target, target_norm = self.evaluate(batch)

        loss = -self.criterion(output, target_norm)

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

    def validation_step(self, batch, batch_idx):
        """
        Calculates various error metrics for validation
        Args:
            batch: Tuple of graph object from pytorch geometric and input for Roost
            batch_idx: identifiers of batch elements
        Returns:
        """
        output, _, pred, target, target_norm = self.evaluate(batch)

        val_loss = -self.criterion(output, target_norm)

        val_mae = mae(pred, target)
        val_rmse = mse(pred, target).sqrt_()
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)
        self.log('val_rmse', val_rmse, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        Calculates various error metrics for testing
        Args:
            batch: Tuple of graph object from pytorch geometric and input for Roost
            batch_idx: identifiers of batch elements
        Returns:
        """
        output, _, pred, target, target_norm = self.evaluate(batch)

        test_loss = self.criterion(output, target_norm)

        test_mae = mae(pred, target)
        test_rmse = mse(pred, target).sqrt_()
        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True)
        self.log('test_mae', test_mae, on_epoch=True, sync_dist=True)
        self.log('test_rmse', test_rmse, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Creates optimizers for training according to the hyperparameter settings
        Args:
        Returns:
            [optimizer], [scheduler]: Tuple of list of optimizers and list of learning rate schedulers
        """
        # Select parameters, which should be trained
        parameters = self.parameters()

        # Select Optimiser
        if self.hparams.optim == "SGD":
            optimizer = optim.SGD(parameters,
                                  lr=self.hparams.learning_rate,
                                  weight_decay=self.hparams.weight_decay,
                                  momentum=self.hparams.momentum)
        elif self.hparams.optim == "Adam":
            optimizer = optim.Adam(parameters,
                                   lr=self.hparams.learning_rate,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim == "AdamW":
            optimizer = optim.AdamW(parameters,
                                    lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)
        else:
            raise NameError(
                "Only SGD, Adam, AdamW are allowed as --optim")

        if self.hparams.clr:
            clr = cyclical_lr(period=self.hparams.clr_period,
                              cycle_mul=0.1,
                              tune_mul=0.05, )
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
        """
        creates dataloader for training according to the hyperparameters
        Args:
        Returns:
            train_generator: Dataloader for training dataset
        """
        params = {"batch_size": self.hparams.batch_size,
                  "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "shuffle": True,
                  "drop_last": True
                  }
        print('length of train_subset: {}'.format(len(self.train_subset)))
        train_generator = DataLoader(
            self.train_subset, collate_fn=collate_fn, **params)
        return train_generator

    def val_dataloader(self):
        """
        creates dataloader for validation according to the hyperparameters
        Args:
        Returns:
            val_generator: Dataloader for validation dataset
        """
        params = {"batch_size": self.hparams.batch_size,
                  #              "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "drop_last": True,
                  "shuffle": False}
        val_generator = DataLoader(
            self.val_subset,
            collate_fn=collate_fn,
            **params)
        print('length of val_subset: {}'.format(len(self.val_subset)))
        return val_generator

    def test_dataloader(self):
        """
        creates dataloader for testing according to the hyperparameters
        Args:
        Returns:
            test_generator: Dataloader for testing dataset
        """
        params = {"batch_size": self.hparams.batch_size,
                  #              "num_workers": self.hparams.workers,
                  "pin_memory": False,
                  "drop_last": True,
                  "shuffle": False}
        test_generator = DataLoader(
            self.test_set,
            collate_fn=collate_fn,
            **params)
        print('length of test_subset: {}'.format(len(self.test_set)))
        return test_generator

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser = None) -> ArgumentParser:  # pragma: no-cover
        """
        Parameters defined here will be available through self.hparams
        Args:
            parent_parser: ArgumentParser from e.g. the training script that adds gpu settings and Trainer settings
        Returns:
            parser: ArgumentParser for all hyperparameters and training/test settings
        """
        if parent_parser is not None:
            parser = ArgumentParser(parents=[parent_parser])
        else:
            parser = ArgumentParser()

        parser.add_argument("--data-path",
                            type=str,
                            default="data/",
                            metavar="PATH",
                            help="path to folder/file that contains dataset files, tries to load all "
                                 "*.pickle.gz in folder")
        parser.add_argument("--fea-path",
                            type=str,
                            default="../embeddings/matscholar-embedding.json",
                            metavar="PATH",
                            help="atom feature path")
        parser.add_argument("--max-nbr",
                            default=24,
                            type=int,
                            metavar="max_N",
                            help="num of neighbors maximum depends on the number set during the feature calculation")
        parser.add_argument("--version",
                            type=str,
                            default="CGAT",
                            help="module from which to load CGAtNet class")
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
        parser.add_argument("--epochs",
                            default=390,
                            type=int,
                            metavar="N",
                            help="number of total epochs to run")
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
        parser.add_argument("--clr",
                            action="store_false",
                            help="use a cyclical learning rate schedule")
        parser.add_argument("--clr-period",
                            default=130,
                            type=int,
                            help="how many epochs per learning rate cycle to perform")
        parser.add_argument("--train-percentage",
                            default=0.0,
                            type=float,
                            help="Percentage of the training data that is used for training (only use to get"
                                 " a training set size vs test error curve")
        parser.add_argument("--seed",
                            default=0,
                            type=int,
                            metavar="N",
                            help="seed for random number generator")
        parser.add_argument("--smoke-test",
                            action="store_true",
                            help="Finish quickly for testing")
        parser.add_argument("--train",
                            action="store_false",
                            help="if set to True datasets will not be loaded to speed up loading of the model")
        parser.add_argument("--target",
                            default="e_above_hull_new",
                            type=str,
                            metavar="str",
                            help="choose the target variable, the dataset dictionary should have a corresponding"
                                 "dictionary structure data['target'][target]")
        parser.add_argument("--test-path",
                            default=None,
                            type=str,
                            help="path to data set with the test set (only used in combination with --val-path)")
        parser.add_argument("--val-path",
                            default=None,
                            type=str,
                            help="path to data set with the validation set (only used in combination with --val-path)")
        parser.add_argument("--inducing-points",
                            default=1000,
                            type=int,
                            help="Number of points used for inducing the approximate Gaussian process.")
        parser.add_argument("--cgat-model",
                            default=None,
                            type=str,
                            required=True,
                            help="Path to CGAT model, which calculates the embeddings.")

        return parser


def main():
    parent_parser = ArgumentParser(add_help=False)

    # argumentparser for the training process
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='number of gpus to use'
    )
    parent_parser.add_argument(
        '--acc_batches',
        type=int,
        default=1,
        help='number of batches to accumulate'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--amp_optimization',
        type=str,
        default='00',
        help='mixed precision format, default 00 (32), 01 mixed, 02 closer to 16'
    )
    parent_parser.add_argument(
        '--first-gpu',
        type=int,
        default=0,
        help='gpu number to use [first_gpu-first_gpu+gpus]'
    )
    parent_parser.add_argument(
        '--ckp',
        type=str,
        default='',
        help='ckp path, if left empty no checkpoint is used'
    )
    parent_parser.add_argument("--test",
                               action="store_true",
                               help="whether to train or test"
                               )
    parent_parser.add_argument("--pretrained-model",
                               type=str,
                               default=None,
                               help='path to checkpoint of pretrained model for transfer learning')

    # each LightningModule defines arguments relevant to it
    parser = GLightningModel.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    print(hparams)

    model = GLightningModel(hparams)

    name = os.path.join("runs", "f-{s}_t-{date:%Y-%m-%d_%H-%M-%S}").format(
        date=datetime.datetime.now(),
        s=hparams.seed)

    # initialize logger
    logger = TensorBoardLogger("tb_logs", name=name)
    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_mae:.3f}',
        dirpath=os.path.join(os.getcwd(), 'tb_logs/', name),
        save_top_k=1,
        verbose=True,
        monitor='val_mae',
        mode='min')
    # prefix='')

    print('the model will train on the following gpus:', [hparams.first_gpu + el for el in range(hparams.gpus)])
    if hparams.ckp == '':
        trainer = Trainer(
            max_epochs=hparams.epochs,
            gpus=[hparams.first_gpu + el for el in range(hparams.gpus)],
            strategy=hparams.distributed_backend,
            amp_backend='apex',
            amp_level=hparams.amp_optimization,
            callbacks=[checkpoint_callback],
            logger=logger,
            check_val_every_n_epoch=2,
            accumulate_grad_batches=hparams.acc_batches,
        )
    else:
        trainer = Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            strategy=hparams.distributed_backend,
            amp_backend='apex',
            amp_level=hparams.amp_optimization,
            callbacks=[checkpoint_callback],
            logger=logger,
            check_val_every_n_epoch=2,
            resume_from_checkpoint=hparams.ckp,
            accumulate_grad_batches=hparams.acc_batches,
        )

    # START TRAINING
    trainer.fit(model)


if __name__ == '__main__':
    main()
