from pytorch_lightning.callbacks import ModelCheckpoint
import os
from argparse import ArgumentParser
import os
import gc
import datetime
import numpy as np

import numpy as np
import torch

import pytorch_lightning as pl
from lightning_module import LightningTemplateModel
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
    model = LightningTemplateModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    name = "runs/f-{s}_t-" + "{date:%d-%m-%Y_%H:%M:%S}".format(
        date=datetime.datetime.now(),
        s=hparams.seed)

    logger = TensorBoardLogger("tb_logs", name=name)
    print([hparams.first_gpu + el for el in range(hparams.gpus)])

    checkpoint_callback = ModelCheckpoint(
        filepath='./tb_logs/' + name + '/chkp.pt',
        save_top_k=15,
        verbose=True,
        monitor='val_mae',
        mode='min',
        prefix='')

    trainer = pl.Trainer(
        early_stop_callback=False,
        max_epochs=hparams.epochs,
        gpus=[hparams.first_gpu + el for el in range(hparams.gpus)],
        distributed_backend=hparams.distributed_backend,
        logger=logger,
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=hparams.ckp,
        weights_summary='full',
        accumulate_grad_batches=hparams.acc_batches,
        gradient_clip_val=0.7,
    )
    print('before training:', model.hparams)
    print('training set length:', len(model.train_subset))
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
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--ckp',
        type=str,
        default='asfasd',
        help='ckp path'
    )

    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    parent_parser.add_argument(
        '--first_gpu',
        type=int,
        default=0,
        help='gpu number to use [first_gpu-first_gpu+gpus]'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(
        parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    print(hyperparams)
    main(hyperparams)
