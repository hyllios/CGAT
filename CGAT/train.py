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
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    name="runs/f-{s}_t-"+"{date:%d-%m-%Y_%H:%M:%S}".format(
                                                date=datetime.datetime.now(),
                                                s=hparams.seed)
                                                
    logger = TensorBoardLogger("tb_logs", name=name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_mae:.2f}',
        dirpath = os.path.join(os.getcwd(),'tb_logs/',name),
        save_top_k=1,
        verbose=True,
        monitor='val_mae',
        mode='min',
        prefix='')
    print([hparams.first_gpu+el for el in range(hparams.gpus)])
    if hparams.ckp=='':
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            distributed_backend=hparams.distributed_backend,
            amp_backend='native',
            amp_level = hparams.amp_optimization,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            check_val_every_n_epoch=2,
    #        auto_scale_batch_size='binsearch',
    #        accumulate_grad_batches=2,
    #        fast_dev_run=True,        
            accumulate_grad_batches = hparams.acc_batches,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            distributed_backend=hparams.distributed_backend,
            amp_backend='native',
            amp_level = hparams.amp_optimization,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            check_val_every_n_epoch=2,
            resume_from_checkpoint=hparams.ckp,
    #        auto_scale_batch_size='binsearch',
    #        accumulate_grad_batches=2,
    #        fast_dev_run=True,        
            accumulate_grad_batches = hparams.acc_batches,
        )


    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=4,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--acc_batches',
        type=int,
        default=1,
        help='gpu number to use [first_gpu-first_gpu+gpus]'
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


    # each LightningModule defines arguments relevant to it
    parser = LightningModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    print(hyperparams)
    main(hyperparams)
