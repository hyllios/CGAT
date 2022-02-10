from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
import glob
from get_additional_data import get_composition
from tqdm import tqdm


def main():
    data_paths = glob.glob(os.path.join("additional_data", "*", "*.pickle.gz"))
    assert len(data_paths) > 0
    print(f"Found {len(data_paths)} datasets")
    model_path = glob.glob(os.path.join("tb_logs",
                                        "runs",
                                        "f-0_t-2022-02-02_12-39-57",
                                        "*.ckpt"))[0]
    model = LightningModel.load(model_path)
    df = pd.DataFrame(columns=['comp', 'mae'])

    for path in tqdm(data_paths):
        dataset = CompositionData(
            data=path,
            fea_path="embeddings/matscholar-embedding.json",
            max_neighbor_number=model.hparams.max_nbr,
            target=model.hparams.target
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
        comp = get_composition(path)
        for batch in loader:
            _, _, pred, target, _ = model.evaluate(batch)
            error = mean_absolute_error(target.cpu().numpy(), target.cpu().numpy())
            df.loc[len(df)] = [comp, error]
    df.to_csv('additional_data/errors.csv', index=False)


if __name__ == '__main__':
    main()
