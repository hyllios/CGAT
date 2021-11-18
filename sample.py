import gzip as gz
import pickle
from pymatgen.core.periodic_table import Element
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Union
from scipy.special import erf
from metropolis import MarkovChain

DIR = 'data'


def getfile(i: int):
    return f'{DIR}/data_{i * 10000}_{(i + 1) * 10000}.pickle.gz'


def find_closest(sample: Union[list, np.ndarray], target):
    # assume sorted list
    if target < sample[0]:
        return 0
    elif target > sample[-1]:
        return len(sample) - 1

    lower_bound = 0
    upper_bound = len(sample) - 1

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        value = sample[mid]

        if target < value:
            upper_bound = mid - 1
        elif target > value:
            lower_bound = mid + 1
        else:
            return mid

    if (sample[lower_bound] - target) < (target - sample[upper_bound]):
        return lower_bound
    else:
        return upper_bound


def find_element(sample: list[set], el: int):
    for i, s in enumerate(sample):
        if el in s:
            return i


def normal_distribution(x):
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)


def cum_distribution(x):
    return (1 + erf(x / np.sqrt(2))) / 2


def get_skewd_gaussion(skew, location, scale):
    def f(x):
        return 2 / scale * normal_distribution((x - location) / scale) * cum_distribution(skew * (x - location) / scale)

    return f


def main():
    spgs = []
    ids = []
    elements = []
    convex_hulls = []

    for i in trange(283):
        data = pickle.load(gz.open(getfile(i), 'rb'))
        for j, d in enumerate(data['batch_ids']):
            split = d[0].split(',')
            spg = int(split[-1].lstrip().rstrip(')'))
            id_ = int(split[0])
            _elements = set([Element(el.rstrip('0123456789')).Z for el in data['batch_comp'][j][0].split()])
            convex_hull = data['target']['e_above_hull_new'][j][0]
            spgs.append(spg)
            ids.append(id_)
            elements.append(_elements)
            convex_hulls.append(convex_hull)

    # indices = [i for i in range(len(spgs))]
    N = 50000
    random.seed(1)
    # sample = random.sample(indices, N)

    plt.hist(spgs, bins=max(spgs), log=True)
    plt.title('Space groups')
    plt.show()

    all_elements = [el for l in elements for el in l]

    plt.hist(all_elements, bins=max(all_elements))
    plt.title('Elements')
    plt.show()

    plt.hist(convex_hulls, bins=100, log=True)
    plt.title('Distance to the convex hull')
    plt.show()

    # spgs = np.array(spgs)
    # elements = np.array(elements)
    # convex_hulls = np.array(convex_hulls)

    # sample for spgs
    # args = np.argsort(spgs)
    # spgs = list(spgs[args])
    # elements = list(elements[args])
    # convex_hulls = list(convex_hulls[args])
    #
    # spgs_sample = []
    # elements_sample = []
    # convex_hulls_sample = []
    # m, M = min(spgs), max(spgs)
    # for _ in trange(N):
    #     spg = random.randint(m, M)
    #     i = find_closest(spgs, spg)
    #     spgs_sample.append(spgs.pop(i))
    #     elements_sample.append(elements.pop(i))
    #     convex_hulls_sample.append(convex_hulls.pop(i))

    # sample for convex hull
    # args = np.argsort(convex_hulls)
    # spgs = list(spgs[args])
    # elements = list(elements[args])
    # convex_hulls = list(convex_hulls[args])
    #
    # spgs_sample = []
    # elements_sample = []
    # convex_hulls_sample = []
    #
    # chain = MarkovChain(get_skewd_gaussion(100, -0.02, 0.7), lambda: random.random() * convex_hulls[-1])
    # chain.step(N - 1)
    # for v in tqdm(chain):
    #     i = find_closest(convex_hulls, v)
    #     spgs_sample.append(spgs.pop(i))
    #     elements_sample.append(elements.pop(i))
    #     convex_hulls_sample.append(convex_hulls.pop(i))

    # sample for elements
    spgs_sample = []
    elements_sample = []
    convex_hulls_sample = []

    element_list = list(set(all_elements))
    for el in tqdm(random.choices(element_list, k=N)):
        i = find_element(elements, el)
        spgs_sample.append(spgs.pop(i))
        elements_sample.append(elements.pop(i))
        convex_hulls_sample.append(convex_hulls.pop(i))

    plt.hist(spgs_sample, bins=max(spgs_sample), log=True)
    plt.title('sampled Space groups')
    plt.show()

    all_elements = [el for l in elements_sample for el in l]

    plt.hist(all_elements, bins=max(all_elements))
    plt.title('sampled Elements')
    plt.show()

    plt.hist(convex_hulls_sample, bins=100, log=True)
    plt.title('sampled Distance to the convex hull')
    plt.show()


if __name__ == '__main__':
    main()
