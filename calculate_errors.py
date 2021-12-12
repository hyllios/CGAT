import gzip
import pickle

import torch

from CGAT.lightning_module import LightningModel, collate_fn
from tqdm import tqdm, trange
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
import sys
from sample import search, get_id


def get_file(i: int, path: str):
    return os.path.join(path, f'data_{i * 10000}_{(i + 1) * 10000}.pickle.gz')


def main():
    parser = LightningModel.add_model_specific_args()

    parser.add_argument(
        '--ckp',
        type=str,
        default='',
        help='ckp path',
        required=True
    )

    parser.add_argument(
        '--idx',
        type=int,
        help='which data file to load: data_{idx*10000}_{(idx+1)*10000}.pickle.gz',
        required=True
    )

    hparams = parser.parse_args()
    i = hparams.idx
    # Disable training for faster loading
    hparams.train = False

    # load model
    model = LightningModel(hparams)
    model.load_from_checkpoint(hparams.ckp, map_location='cpu')
    model.eval()
    model.to('cuda')

    # declare dataframe for saving errors
    errors = pd.DataFrame(columns=['batch_ids', 'errors', 'indices'])
    PATH = 'active_learning'
    UNPREPARED_PATH = 'unprepared_volume_data'
    # iterate over unused data and evaluate the error
    dataset = CompositionData(
        data=get_file(i, PATH),
        fea_path=hparams.fea_path,
        max_neighbor_number=hparams.max_nbr,
        target=hparams.target
    )
    data = pickle.load(gzip.open(get_file(i, PATH), 'rb'))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for j, batch in tqdm(enumerate(loader)):
        _, _, pred, target, _ = model.evaluate(batch)
        row = {'errors': mean_absolute_error(target.cpu().numpy(), pred.cpu().numpy()),
               'batch_ids': data['batch_ids'][j][0]}
        errors = errors.append(row, ignore_index=True)

    errors.to_csv(get_file(i, PATH + '/temp').replace('data', 'errors').replace('pickle.gz', 'csv'))

    # N = 25000
    # errors = errors.sort_values(by='errors', ascending=False, ignore_index=True).head(N)
    # batch_ids = set(errors['batch_ids'].to_list())
    #
    # new_sample = []
    # for i in trange(283):
    #     data = pickle.load(gzip.open(get_file(i, PATH), 'rb'))
    #     unprepared_data = pickle.load(gzip.open(get_file(i, UNPREPARED_PATH), 'rb'))
    #     indices_to_remove = []
    #     current_batch_ids = []
    #     # find all batch_ids for new sample in current file
    #     for j, batch_id in enumerate(data['batch_ids']):
    #         batch_id = batch_id[0]
    #         if batch_id in batch_ids:
    #             indices_to_remove.append(j)
    #             current_batch_ids.append(get_id(batch_id))
    #             batch_ids.remove(batch_id)
    #
    #     if len(indices_to_remove) > 0:
    #         # reverse order of indices for easy popping
    #         indices_to_remove.reverse()
    #         # remove those entries from data
    #         data['input'] = np.delete(data['input'], indices_to_remove, axis=1)
    #         for j in indices_to_remove:
    #             data['batch_ids'].pop(j)
    #         data['batch_comp'] = np.delete(data['batch_comp'], indices_to_remove)
    #         data['comps'] = np.delete(data['comps'], indices_to_remove)
    #         for target in data['target']:
    #             data['target'][target] = np.delete(data['target'][target], indices_to_remove)
    #         pickle.dump(data, gzip.open(get_file(i, PATH), 'wb'))
    #
    #         # find batch_ids in unprepared data and append to new_sample
    #         for batch_id in current_batch_ids:
    #             new_sample.append(unprepared_data.pop(search(unprepared_data, batch_id)))
    # # save new sample
    # pickle.dump(new_sample, gzip.open('./active_learning/sample/unprepared_sample_1.pickle.gz', 'wb'))


if __name__ == '__main__':
    main()
