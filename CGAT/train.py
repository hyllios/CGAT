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
from .lightning_module import LightningModel
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # initialize model
    if hparams.pretrained_model is None:
        model = LightningModel(hparams)
    else:
        assert os.path.isfile(hparams.pretrained_model), f"Checkpoint file {hparams.pretrained_model} does not exist!"
        # load model from checkpoint and override old hyperparameters
        model = LightningModel.load_from_checkpoint(hparams.pretrained_model, **vars(hparams))
    # definte path for model checkpoints and tensorboard 
    name = "runs/f-{s}_t-{date:%Y-%m-%d_%H-%M-%S}".format(
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
        trainer = pl.Trainer(
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
        trainer = pl.Trainer(
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


def run():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # argumentparser for the training process
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=4,
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
    parser = LightningModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    print(hyperparams)
    main(hyperparams)


if __name__ == '__main__':
    run()
