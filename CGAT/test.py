from pytorch_lightning.callbacks import ModelCheckpoint
import os
from argparse import ArgumentParser
import os
import gc
import datetime
import numpy as np
import pandas as pd

import numpy as np
import torch

import pytorch_lightning as pl
from lightning_module import LightningModel
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    testing routine
    Args:
        hparams: checkpoint of the model to be tested and gpu, parallel backend etc.,
                 defined in the argument parser in if __name__ == '__main__':
    Returns:
    """
    checkpoint_path=hparams.ckp
    model = LightningModel.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    tags_csv= hparams.hparams,
    )

    trainer = pl.Trainer(
        gpus=[hparams.first_gpu+el for el in range(hparams.gpus)],
        distributed_backend=hparams.distributed_backend,
    )

    trainer.test(model)

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=4,
        help='how many gpus'
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
        help="mixed precision format, default 00 (32), 01 mixed, 02 closer to 16, should not be used during testing"
    )
    parent_parser.add_argument(
        '--first-gpu',
        type=int,
        default=0,
        help='gpu number to use [first_gpu, ..., first_gpu+gpus]'
    )
    parent_parser.add_argument(
        '--ckp',
        type=str,
        default='',
        help='ckp path, if left empty no checkpoint is used'
    )
    parent_parser.add_argument(
        '--hparams',
        type=str,
        default='',
        help='path for hparams of ckp if left empty no checkpoint is used'
    )
    parent_parser.add_argument("--test",
        action="store_true",
        help="whether to train or test"
    )


    # each LightningModule defines arguments relevant to it
    parser = LightningModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    print(hyperparams)
    main(hyperparams)
