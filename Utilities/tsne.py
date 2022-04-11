import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from openTSNE import TSNE
from tqdm import tqdm


def cap(array, upper_limit):
    temp = np.full((len(array), 2), upper_limit)
    temp[:, 0] = array
    return np.amin(temp, axis=1)


def main():
    comps = Path('new_active_learning').glob('A*B*')

    embeddings = []
    errors = []
    colors = []
    targets = []
    comp_id = []

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, comp in enumerate(tqdm(list(comps))):
        files = comp.glob('*.txt')
        predictions = pd.DataFrame()
        for file in files:
            if 'target' == file.stem:
                targets.append(np.loadtxt(file))
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
        errors.append(np.array([mean_absolute_error([targets[-1][i]] * len(row.keys()), row) for i, row in
                                predictions.iterrows()]
                               ))
        colors += [cycle[i % len(cycle)]] * len(errors[-1])
        comp_id += [i] * len(errors[-1])

    embeddings = np.concatenate(embeddings)
    errors = np.concatenate(errors)
    targets = np.concatenate(targets)

    errors = cap(errors, .4)
    targets = cap(targets, .5)

    tsne = TSNE(n_jobs=6, perplexity=300)
    embedding = tsne.fit(embeddings)

    plt.title('Error')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=errors, s=.5)
    plt.colorbar()
    plt.figure()
    plt.title('Prototypes')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=.5)
    plt.show()
    plt.figure()
    plt.title('distance to convex hull')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, s=.5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
