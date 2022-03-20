from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
import glob
from get_additional_data import get_composition
from tqdm import tqdm
import numpy as np


def main():
    data_paths = glob.glob(os.path.join("additional_data", "*", "*.pickle.gz"))
    assert len(data_paths) > 0
    print(f"Found {len(data_paths)} datasets")
    # sizes = [50_000, 75_000, 100_000, 125_000, 150_000, 200_000, 250_000]
    # runs = ["f-0_t-2022-01-03_15-07-47",
    #         "f-0_t-2022-01-09_14-33-04",
    #         "f-0_t-2022-01-14_14-50-14",
    #         "f-0_t-2022-01-18_10-00-20",
    #         "f-0_t-2022-01-22_17-40-56",
    #         "f-0_t-2022-01-26_22-42-02",
    #         "f-0_t-2022-02-02_12-39-57"]
    # model_paths = [glob.glob(os.path.join("tb_logs",
    #                                       "runs",
    #                                       "{run}",
    #                                       "*.ckpt").format(run=run))[0] for run in runs]
    model_paths = glob.glob(os.path.join('new_active_learning', 'checkpoints', '*', '*.ckpt'))
    df = pd.DataFrame(columns=['comp', 'run', 'mae'])
    for i, model_path in enumerate(tqdm(model_paths)):
        model = LightningModel.load(model_path)

        for path in tqdm(data_paths):
            dataset = CompositionData(
                data=path,
                fea_path="embeddings/matscholar-embedding.json",
                max_neighbor_number=model.hparams.max_nbr,
                target=model.hparams.target
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
            comp = get_composition(path)
            errors = []
            for batch in loader:
                _, _, pred, target, _ = model.evaluate(batch)
                errors.append(mean_absolute_error(target.cpu().numpy(), pred.cpu().numpy()))
            df.loc[len(df)] = [comp, i, np.mean(errors)]
    df.to_csv('additional_data/errors.csv', index=False)


if __name__ == '__main__':
    main()
