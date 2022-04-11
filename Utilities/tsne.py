import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from fastTSNE import TSNE
from tqdm import tqdm


def main():
    comps = Path('new_active_learning').glob('A*B*')

    embeddings = []
    errors = []
    for comp in tqdm(list(comps)):
        files = comp.glob('*.txt')
        predictions = pd.DataFrame()
        for file in files:
            if 'target' == file.stem:
                targets = np.loadtxt(file)
                # targets.append(target)
            elif 'log_std' in file.stem:
                pass
            elif 'embeddings' == file.stem:
                embeddings.append(np.loadtxt(file))
            else:
                try:
                    predictions[int(file.stem)] = np.loadtxt(file)
                except ValueError:
                    pass
        errors.append(np.array([mean_absolute_error([targets[i]] * len(row.keys()), row) for i, row in
                                predictions.iterrows()]
                               ))

    embeddings = np.concatenate(embeddings)
    errors = np.concatenate(errors)

    print(embeddings.shape)
    tsne = TSNE()
    embedding = tsne.fit(embeddings)
    print(embedding.shape)


if __name__ == '__main__':
    main()
