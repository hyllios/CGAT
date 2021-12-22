import gzip
import pickle

from CGAT.lightning_module import LightningModel, collate_fn
from tqdm import trange
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error


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

    # parser.add_argument(
    #     '--idx',
    #     type=int,
    #     help='which data file to load: data_{idx*10000}_{(idx+1)*10000}.pickle.gz',
    #     required=True
    # )

    hparams = parser.parse_args()
    # i = hparams.idx
    # Disable training for faster loading
    hparams.train = False

    # load model
    # model = LightningModel(hparams)
    # model.load_from_checkpoint(hparams.ckp)
    # model.eval()
    # model.to('cuda')
    model = LightningModel.load(hparams.ckp)

    # declare dataframe for saving errors
    errors = pd.DataFrame(columns=['batch_ids', 'errors'])
    PATH = 'active_learning'
    # iterate over unused data and evaluate the error
    for i in trange(283):
        dataset = CompositionData(
            data=get_file(i, PATH),
            fea_path=hparams.fea_path,
            max_neighbor_number=hparams.max_nbr,
            target=hparams.target
        )
        data = pickle.load(gzip.open(get_file(i, PATH), 'rb'))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        for j, batch in enumerate(loader):
            _, _, pred, target, _ = model.evaluate(batch)
            row = {'errors': mean_absolute_error(target.cpu().numpy(), pred.cpu().numpy()),
                   'batch_ids': data['batch_ids'][j][0]}
            errors = errors.append(row, ignore_index=True)

        errors.to_csv(get_file(i, PATH + '/temp').replace('data', 'errors').replace('pickle.gz', 'csv'))


if __name__ == '__main__':
    main()
