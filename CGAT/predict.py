from lightning_module import LightningModel
from data import CompositionData
import torch
from torch.utils.data import DataLoader
import pickle, gzip as gz


model_path = '/home/test_user/QMC/test_CGAT/CGAT/CGAT/tb_logs/runs/f-{s}_t-11-10-2021_09:14:23/epoch=3-val_mae=0.19.ckpt'
model = LightningModel.load_from_checkpoint(model_path, train=False)

data_path = 'test_data/layered_perovskites_input.pickle.gz'
dataset = CompositionData(
    data=data_path,
    fea_path=model.hparams.fea_path,
    max_neighbor_number=model.hparams.max_nbr)


params = {"batch_size": 5000,
          "pin_memory": False,
          "shuffle": False,
          "drop_last": False
          }

def collate_fn(datalist):
    return datalist


model = model.cuda()
dataloader = DataLoader(dataset, collate_fn=collate_fn, **params)

prediction_list = []
for i, batch in enumerate(dataloader):
    with torch.no_grad():
        print(i)
        prediction_list.append(model.evaluate(batch)[2]) # tuple of size 5 output, log_std, pred, target, target_norm

        
pickle.dump(
    torch.cat(prediction_list),
    gz.open(
        'predictions.pickle.gz',
        'wb'))

