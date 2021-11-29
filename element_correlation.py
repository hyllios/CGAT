import pickle
import gzip as gz
import matplotlib.pyplot as plt
import numpy as np
from sample import getfile
from tqdm import trange, tqdm
from pymatgen.core.periodic_table import Element
import re


def get_distribution(hist: np.ndarray):
    def f(x):
        x = int(x)
        return hist[x]

    return f


def main():
    elements: list[list[int]] = []
    pattern = re.compile(r'([a-zA-Z]+)\d+')
    for i in trange(283):
        data = pickle.load(gz.open(getfile(i), 'rb'))
        for comp, in data['batch_comp']:
            elements.append([Element(el).Z for el in pattern.findall(comp)])
    biggest_element = max([max(els) for els in elements])
    correlation_matrix = np.zeros((biggest_element, biggest_element))
    for els in tqdm(elements):
        for i in els:
            for j in els:
                correlation_matrix[i - 1, j - 1] += 1

    correlation_matrix = (
            correlation_matrix.T / np.where(correlation_matrix.diagonal() != 0, correlation_matrix.diagonal(),
                                            np.ones(biggest_element))).T

    for i in range(biggest_element):
        correlation_matrix[i, i] = 0
    plt.matshow(correlation_matrix)
    plt.colorbar()
    print(np.sort(correlation_matrix.flatten())[:-10:-1])

    plt.figure()
    x = np.linspace(0, biggest_element, biggest_element * 100, endpoint=False)
    # plt.hist(list(range(biggest_element)), bins=biggest_element, weights=correlation_matrix.mean(axis=0))
    f = get_distribution(correlation_matrix.mean(axis=0))
    y = np.array([f(i) for i in x])
    plt.plot(x, np.where(y != 0, y, np.zeros_like(y)))
    plt.figure()
    plt.plot(x, [min(i, 200) for i in np.where(y > 1e-3, 1 / y, np.zeros_like(y))])

    plt.show()


if __name__ == '__main__':
    main()
