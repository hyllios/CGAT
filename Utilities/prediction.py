import torch

from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from get_additional_data import get_composition
from tqdm import tqdm
import re


def get_seed(path):
    pattern = re.compile(r'f-(\d+)_')
    return int(pattern.search(path).group(1))


def main():
    data_paths = glob.glob(os.path.join("additional_data", "*", "*.pickle.gz"))
    assert len(data_paths) > 0
    print(f"Found {len(data_paths)} datasets")
    model_paths = sorted(glob.glob(os.path.join('new_active_learning', 'checkpoints', 'e_hull', '350_000', '*', '*.ckpt')),
                         key=get_seed)
    seeds = list(map(get_seed, model_paths))
    # df = pd.DataFrame(columns=['comp', 'seed', 'entry', 'prediction'])
    for seed, model_path in zip(seeds, tqdm(model_paths)):
        model = LightningModel.load_from_checkpoint(model_path, train=False)
        model = model.cuda()

        for path in tqdm(data_paths):
            dataset = CompositionData(
                data=path,
                fea_path="embeddings/matscholar-embedding.json",
                max_neighbor_number=model.hparams.max_nbr,
                target=model.hparams.target
            )
            loader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)
            comp = get_composition(path)
            predictions = []
            targets = []
            log_stds = []
            for batch in loader:
                with torch.no_grad():
                    _, log_std, pred, target, _ = model.evaluate(batch)
                predictions.append(pred)
                targets.append(target)
                log_stds.append(log_std)
            dir = os.path.join('new_active_learning', comp)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            np.savetxt(os.path.join(dir, f'{seed}.txt'), torch.cat(predictions).cpu().numpy().reshape((-1,)))
            np.savetxt(os.path.join(dir, f'target.txt'), torch.cat(targets).cpu().numpy().reshape((-1,)))
            np.savetxt(os.path.join(dir, f'log_std_{seed}.txt'), torch.cat(log_stds).cpu().numpy().reshape((-1,)))


if __name__ == '__main__':
    main()
