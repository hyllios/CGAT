from lightning_module import LightningModel
from data import CompositionData
import torch
from torch.utils.data import DataLoader
import pickle, gzip as gz
from pytorch_lightning import Trainer
torch.set_printoptions(6)
#model_path = 'tb_logs/runs/f-{s}_t-30-11-2021_12:27:59/epoch=261-val_mae=0.24.ckpt'
model_path = 'tb_logs/runs/f-{s}_t-25-11-2021_21:10:33/epoch=261-val_mae=0.02.ckpt'
model = LightningModel.load_from_checkpoint(model_path, train=False)

#data_path = 'test_all_same_new.pickle.gz'
data_path = '../../data_noah/data_2220000_2230000.pickle.gz'
dataset = CompositionData(
    data=data_path,
    fea_path=model.hparams.hparams["fea_path"],
    max_neighbor_number=model.hparams.hparams["max_nbr"], target='e_above_hull_new')
print(len(dataset))

params = {"batch_size": 234,
          "pin_memory": False,
          "shuffle": False,
          "drop_last": False
          }

def collate_fn(datalist):
    return datalist


model = model.cuda()
dataloader = DataLoader(dataset, collate_fn=collate_fn, **params)


trainer = Trainer(gpus=10, strategy='ddp')
prediction_list = trainer.predict(model, dataloader)

#print(torch.mean(torch.cat(prediction_list)))
print(prediction_list)        
pickle.dump(prediction_list,gz.open('new_predict.pickle.gz','wb'))

