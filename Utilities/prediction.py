from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
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
    model_paths = sorted(glob.glob(os.path.join('new_active_learning', 'checkpoints', '*', '*.ckpt')), key=get_seed)
    seeds = list(map(get_seed, model_paths))
    df = pd.DataFrame(columns=['comp', 'seed', 'entry', 'prediction'])
    for i, model_path in zip(seeds, tqdm(model_paths)):
        model = LightningModel.load_from_checkpoint(model_path, train=False)
        model = model.cuda()

        for path in tqdm(data_paths):
            dataset = CompositionData(
                data=path,
                fea_path="embeddings/matscholar-embedding.json",
                max_neighbor_number=model.hparams.max_nbr,
                target=model.hparams.target
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
            comp = get_composition(path)
            for j, batch in enumerate(loader):
                _, _, pred, target, _ = model.evaluate(batch)
                df.loc[len(df)] = [comp, i, j, pred.cpu()]
        df.to_csv('new_active_learning/predictions.csv', index=False)


if __name__ == '__main__':
    main()
