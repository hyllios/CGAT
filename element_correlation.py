import pickle
import gzip as gz

import matplotlib.pyplot as plt
import numpy as np

from sample import getfile
from tqdm import trange, tqdm
from pymatgen.core.periodic_table import Element
import re


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
    print(np.sort(correlation_matrix.flatten())[:-10:-1])
    plt.show()


if __name__ == '__main__':
    main()
