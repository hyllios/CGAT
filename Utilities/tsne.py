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


def plot(embedding, value, **kwargs):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=value, **kwargs)
    plt.colorbar()


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
            elif 'graph_embeddings' == file.stem:
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

    titles = ['Error', 'Prototypes', 'distance to convex hull']
    values = [errors, colors, targets]

    embedding_list = []
    for metric in ['euclidean', 'cosine']:
        tsne = TSNE(n_jobs=6, perplexity=500, metric=metric, exaggeration=2)
        embedding = tsne.fit(embeddings)
        embedding_list.append(embedding)

        for title, value in zip(titles, values):
            plt.figure()
            plt.title(f'{title} -- {metric}')
            plot(embedding, value, s=.5)
        plt.show()

    embedding = embedding_list[0]
    for e in embedding_list[1:]:
        embedding *= e
    for title, value in zip(titles, values):
        plt.figure()
        plt.title(f'{title} -- product')
        plot(embedding, value, s=.5)
    plt.show()


if __name__ == '__main__':
    main()
