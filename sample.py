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
from pymatgen.entries.computed_entries import ComputedStructureEntry

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


def get_distribution(hist: Union[np.ndarray, list]):
    def f(x):
        x = int(x)
        return hist[x]

    return f


def get_id(entry: Union[ComputedStructureEntry,str] ) -> int:
    if isinstance(entry, ComputedStructureEntry):
        return int(entry.data['id'].split(',')[0])
    elif isinstance(entry, str):
        return int(entry.split(',')[0])


def search(data: list[ComputedStructureEntry], batch_id: int):
    low = 0
    high = len(data) - 1

    while low <= high:
        mid = (high + low) // 2
        curr_id = get_id(data[mid])
        if curr_id < batch_id:
            low = mid + 1
        elif curr_id > batch_id:
            high = mid - 1
        else:
            return mid
    raise ValueError


def main():
    batch_ids = []
    elements = []
    stoichiometries = []

    test_val_data = pickle.load(gz.open('data/test_and_val_idx.pickle.gz', 'rb'))
    test_batch_ids = set([batch_id[0] for batch_id in test_val_data['test_batch_ids']])
    val_batch_ids = set([batch_id[0] for batch_id in test_val_data['val_batch_ids']])
    test_val_batch_ids = test_batch_ids | val_batch_ids

    # files = sorted([getfile(i) for i in range(283)])

    for i in trange(283):
        data = pickle.load(gz.open(getfile(i), 'rb'))
        for j, d in enumerate(data['batch_ids']):
            if d[0] not in test_val_batch_ids:
                split = d[0].split(',')
                # batch_id = int(split[0])
                _elements = set([Element(el.rstrip('0123456789')).Z for el in data['batch_comp'][j][0].split()])
                batch_ids.append(d[0])
                elements.append(_elements)
                stoichiometries.append(data['batch_comp'][j][0])

    print(f'{len(set(stoichiometries)) / len(stoichiometries):.2%} unique stoichiometries')

    print('Calculating correlation matrix')
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

    y = correlation_matrix.mean(axis=0)
    distribution = get_distribution([min(150, i) for i in np.where(y > 1e-3, 1 / y, np.zeros_like(y))])

    # indices = [i for i in range(len(spgs))]
    N = 50000
    random.seed(1)
    # sample = random.sample(indices, N)

    # all_elements = [el for l in elements for el in l]
    #
    # plt.figure()
    # plt.hist(all_elements, bins=max(all_elements))
    # plt.title('Elements')
    # plt.savefig('elements.pdf')
    # plt.show()

    batch_ids = np.array(batch_ids)
    elements = np.array(elements)
    stoichiometries = np.array(stoichiometries)

    np.random.seed(0)
    args = [i for i in range(len(batch_ids))]
    np.random.shuffle(args)
    batch_ids = list(batch_ids[args])
    elements = list(elements[args])
    stoichiometries = list(stoichiometries[args])

    # sample for elements
    # sample_batch_ids = set()
    # elements_sample = []
    # stoichiometries_sample = set()
    #
    # chain = MarkovChain(distribution, lambda: random.randint(0, biggest_element - 1))
    # chain.step(N)
    #
    # # element_list = list(set(all_elements))
    # # for el in tqdm(random.choices(element_list, k=N)):
    # bar = tqdm(total=N)
    # while len(sample_batch_ids) < N:
    #     chain.step(1)
    #     el = chain[-1] + 1
    #     while True:
    #         i = find_element(elements, el)
    #         if i is None:
    #             break
    #         stoichiometry = stoichiometries.pop(i)
    #         if stoichiometry not in stoichiometries_sample:
    #             sample_batch_ids.add(batch_ids.pop(i))
    #             elements_sample.append(elements.pop(i))
    #             stoichiometries_sample.add(stoichiometry)
    #             bar.update(1)
    #             break
    #         else:
    #             elements.pop(i)
    #             batch_ids.pop(i)
    # bar.close()

    # random sample
    sample_batch_ids = set(random.sample(batch_ids, N))

    # all_elements = [el for l in elements_sample for el in l]
    #
    # plt.figure()
    # plt.hist(all_elements, bins=max(all_elements))
    # plt.title('sampled Elements')
    # plt.savefig('sampled_elements.pdf')
    # plt.show()
    sample_data = []
    test_data = []
    val_data = []
    for i in trange(283):
        data = pickle.load(gz.open(getfile(i), 'rb'))
        sample_indices = []
        test_val_indices = []
        curr_sample_batch_ids = []
        curr_test_batch_ids = []
        curr_val_batch_ids = []
        for j, batch_id in enumerate(data['batch_ids']):
            batch_id = batch_id[0]
            if batch_id in sample_batch_ids:
                sample_indices.append(j)
                sample_batch_ids.remove(batch_id)
                curr_sample_batch_ids.append(get_id(batch_id))
            elif batch_id in test_batch_ids:
                test_val_indices.append(j)
                test_val_batch_ids.remove(batch_id)
                test_batch_ids.remove(batch_id)
                curr_test_batch_ids.append(get_id(batch_id))
            elif batch_id in val_batch_ids:
                test_val_indices.append(j)
                test_val_batch_ids.remove(batch_id)
                val_batch_ids.remove(batch_id)
                curr_val_batch_ids.append(get_id(batch_id))
        if len(sample_indices) > 0 or len(test_val_indices) > 0:
            unprepared_data: list[ComputedStructureEntry] = pickle.load(
                gz.open(getfile(i).replace(f'{DIR}/', 'unprepared_volume_data/')))
            for batch_id in curr_sample_batch_ids:
                j = search(unprepared_data, batch_id)
                sample_data.append(unprepared_data.pop(j))
            for batch_id in curr_test_batch_ids:
                j = search(unprepared_data, batch_id)
                test_data.append(unprepared_data.pop(j))
            for batch_id in curr_val_batch_ids:
                j = search(unprepared_data, batch_id)
                val_data.append(unprepared_data.pop(j))
            # if len(sample_data.keys()) == 0:
            #     sample_data['input'] = data['input'][:, sample_indices]
            #     sample_data['batch_ids'] = [data['batch_ids'][j] for j in sample_indices]
            #     sample_data['batch_comp'] = data['batch_comp'][sample_indices]
            #     sample_data['target'] = {}
            #     for target in data['target']:
            #         sample_data['target'][target] = data['target'][target][sample_indices]
            #     sample_data['comps'] = data['comps'][sample_indices]
            # else:
            #     sample_data['input'] = np.concatenate((sample_data['input'], data['input'][:, sample_indices]), axis=1)
            #     sample_data['batch_ids'] += [data['batch_ids'][j] for j in sample_indices]
            #     sample_data['batch_comp'] = np.concatenate((sample_data['batch_comp'], data['batch_comp']))
            #     for target in data['target']:
            #         sample_data['target'][target] = np.concatenate((sample_data['target'][target],
            #                                                         data['target'][target][sample_indices]))
            #     sample_data['comps'] = np.concatenate((sample_data['comps'], data['comps'][sample_indices]))
            all_used_indices = sorted(sample_indices + test_val_indices, reverse=True)
            data['input'] = np.delete(data['input'], all_used_indices, axis=1)
            for j in all_used_indices:
                data['batch_ids'].pop(j)
            data['batch_comp'] = np.delete(data['batch_comp'], all_used_indices)
            data['comps'] = np.delete(data['comps'], all_used_indices)
            for target in data['target']:
                data['target'][target] = np.delete(data['target'][target], all_used_indices)
        pickle.dump(data, gz.open(getfile(i).replace(f'{DIR}/', 'active_learning/'), 'wb'))
    pickle.dump(sample_data, gz.open('active_learning/unprepared_random_sample.pickle.gz', 'wb'))
    pickle.dump(test_data, gz.open('active_learning/unprepared_test_data.pickle.gz', 'wb'))
    pickle.dump(val_data, gz.open('active_learning/unprepared_val_data.pickle.gz', 'wb'))




if __name__ == '__main__':
    main()
