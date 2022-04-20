import torch
from CGAT.gaussian_process import GLightningModel, EmbeddingData
from glob import glob
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
import numpy as np


def main():
    data_paths = glob(os.path.join("new_active_learning", "A*B*", "graph_embeddings.txt"))
    assert len(data_paths) > 0
    print(f"Found {len(data_paths)} datasets.")

    model_path = os.path.join("new_active_learning", "gp.ckpt")
    model = GLightningModel.load_from_checkpoint(model_path, train=False)

    for path in tqdm(data_paths):
        dataset = EmbeddingData(path, model.hparams.target)
        loader = DataLoader(dataset, batch_size=500, shuffle=False)
        predictions = []
        uncertainties = []
        errors = []
        for batch in loader:
            with torch.no_grad():
                _, (_, upper), pred, target, _ = model.evaluate(batch)
                predictions.append(pred.numpy())
                uncertainties.append(upper.numpy() - pred.numpy())
                errors.append(np.abs(pred.numpy() - target.numpy()))
        df = pd.DataFrame()
        columns = ['prediction', 'uncertainty', 'absolute error']
        lists = [predictions, uncertainties, errors]
        for column, list in zip(columns, lists):
            df[column] = np.concatenate(list)
        df.to_csv(os.path.join(os.path.dirname(path), 'gp_results.csv'), index=False)



if __name__ == '__main__':
    main()
