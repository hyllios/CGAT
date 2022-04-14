import pickle
import gzip as gz
from argparse import ArgumentParser
from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import os
from glob import glob
import torch
from tqdm import tqdm


def load(file):
    return pickle.load(gz.open(file))


def save(data, file):
    pickle.dump(data, gz.open(file, 'wb'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--data-path', '-d',
                        type=str,
                        required=True)
    parser.add_argument('--target-path', '-t',
                        type=str,
                        required=True)
    parser.add_argument('--model-path', '-m',
                        type=str,
                        required=True)
    parser.add_argument('--fea-path', '-f',
                        type=str,
                        default=None)
    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=100)
    args = parser.parse_args()

    model = LightningModel.load_from_checkpoint(args.model_path, train=False)
    model.cuda()

    if os.path.isdir(args.data_path):
        files = glob(os.path.join(args.data_path, '*.pickle.gz'))
    else:
        files = [args.data_path]

    if os.path.isfile(args.target_path):
        raise ValueError("'target-path' must not be a existing file!")

    if not os.path.isdir(args.target_path):
        os.makedirs(args.target_path)

    for file in tqdm(files):
        data = load(file)
        dataset = CompositionData(
            data=data,
            fea_path=args.fea_path if args.fea_path else model.hparams.fea_path,
            max_neighbor_number=model.hparams.max_nbr,
            target=model.hparams.target
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        embedding_list = []
        for batch in loader:
            with torch.no_grad():
                embedding_list.append(model.evaluate(batch, return_graph_embedding=True).cpu())
        embedding = torch.cat(embedding_list).numpy()
        data['input'] = embedding
        save(data, os.path.join(args.target_path, os.path.basename(file)))


if __name__ == '__main__':
    main()
