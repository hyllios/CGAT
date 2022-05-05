import gzip
import pickle

from CGAT.lightning_module import LightningModel, collate_fn
from tqdm import trange
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from pytorch_lightning import Trainer


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
        '--gpus',
        type=int,
        default=2,
        help='number of gpus to use'
    )
    parser.add_argument(
        '--acc_batches',
        type=int,
        default=1,
        help='number of batches to accumulate'
    )
    parser.add_argument(
        '--distributed_backend',
        type=str,
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    parser.add_argument(
        '--amp_optimization',
        type=str,
        default='00',
        help='mixed precision format, default 00 (32), 01 mixed, 02 closer to 16'
    )

    hparams = parser.parse_args()
    # Disable training for faster loading
    hparams.train = False

    # load model
    model = LightningModel.load(hparams.ckp)

    trainer = Trainer(
        gpus=hparams.gpus,
        strategy=hparams.distributed_backend,
        amp_backend='apex',
        amp_level=hparams.amp_optimization,
        accumulate_grad_batches=hparams.acc_batches,
    )

    PATH = 'active_learning'
    # iterate over unused data and evaluate the error
    for i in trange(283):
        # declare dataframe for saving errors
        errors = pd.DataFrame(columns=['batch_ids', 'errors'])
        dataset = CompositionData(
            data=get_file(i, PATH),
            fea_path=hparams.fea_path,
            max_neighbor_number=hparams.max_nbr,
            target=hparams.target
        )
        data = pickle.load(gzip.open(get_file(i, PATH), 'rb'))
        targets = data['target'][hparams.target].reshape((-1, 1, 1))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        # TODO get prediction from other GPU!
        predictions = trainer.predict(model=model, dataloaders=loader)
        for j, batch in enumerate(predictions):
            row = {'errors': mean_absolute_error(targets[j], predictions[j].cpu().numpy()),
                   'batch_ids': data['batch_ids'][j][0]}
            errors = errors.append(row, ignore_index=True)

        errors.to_csv(get_file(i, PATH + '/temp').replace('data', 'errors').replace('pickle.gz', 'csv'), index=False)


if __name__ == '__main__':
    main()
