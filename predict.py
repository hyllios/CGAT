from torch.utils.data import DataLoader
from roost_message import collate_batch
import pickle
import gzip as gz
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
import importlib
import pytorch_lightning as pl
from lightning_module import LightningTemplateModel
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch_geometric.data import Batch
from data import CompositionData
from torch_geometric.data import DataListLoader
import sys
start = int(sys.argv[1])
file = sys.argv[2]

# path to model
model = LightningTemplateModel.load_from_checkpoint('path.ckpt')
dataset = CompositionData(
    data_path=file,
    fea_path=model.hparams.fea_path,
    target_property=model.hparams.target_property,
    max_neighbor_number=model.hparams.max_nbr,
    start=start)
params = {"batch_size": 5000,
          "pin_memory": False,
          "shuffle": False,
          "drop_last": False
          }
print('length of train_subset', len(dataset))


def collate_fn(datalist):
    return datalist


model = model.cuda()

model.eval()
dir = 'already_calculated/'
files = [
    'full_data_ml_f.pickle.gz',
    'full_data_ml_0.85_f.pickle.gz',
    'full_data_ml_0.9_f.pickle.gz',
    'full_data_ml_1.05_f.pickle.gz',
    'full_data_ml_1.1_f.pickle.gz']
files = [dir + el for el in files]
datasets = []
for file in files:
    datasets.append(
        CompositionData(
            data_path=file,
            fea_path=model.hparams.fea_path,
            target_property=model.hparams.target_property,
            max_neighbor_number=model.hparams.max_nbr,
            start=0))

prediction_list_big = []
for dataset in datasets:
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **params)
    prediction_list = []
    model.eval()
    for i, batch in enumerate(dataloader):
        print(i)
        with torch.no_grad():
            device = next(model.model.parameters()).device
            b_comp, batch = [el[1] for el in batch], [el[0] for el in batch]
            batch = (Batch.from_data_list(batch)).to(device)
            b_comp = collate_batch(b_comp)
            b_comp = (tensor.to(device) for tensor in b_comp)
            target = batch.y.view(len(batch.y), 1)
            output, log_std = model(batch, b_comp).chunk(2, dim=1)
            target_norm = model.norm(target)
            prediction_list.append(
                output * model.std.cuda() + model.mean.cuda())
    prediction_list = torch.cat(prediction_list).flatten()
    prediction_list_big.append(prediction_list)

pickle.dump(
    torch.stack(prediction_list_big),
    gz.open(
        'predictions' +
        file2 +
        'pickle.gz',
        'wb'))
