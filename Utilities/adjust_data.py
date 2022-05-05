from typing import Union
import pickle
import gzip as gz
from pymatgen.entries.computed_entries import ComputedStructureEntry
import numpy as np
from tqdm import tqdm, trange
import os


def get_batch_ids(path: Union[str, list[str]]) -> set:
    if isinstance(path, str):
        data = load(path)
        batch_ids = [batch_id[0] for batch_id in data['batch_ids']]
    elif isinstance(path, list):
        batch_ids = []
        for p in tqdm(path):
            data = load(p)
            batch_ids.extend((batch_id[0] for batch_id in data['batch_ids']))
    else:
        raise TypeError("Argument 'path' has to be either a string or list of strings.")
    return set(batch_ids)


def remove_batch_ids(data: dict, batch_ids: set, inplace: bool = True, modify_batch_ids: bool = True) -> dict:
    if len(batch_ids) == 0:
        return data
    if not modify_batch_ids:
        batch_ids = batch_ids.copy()
    # create list of indices which have to be removed
    indices_to_remove = []
    for i, (batch_id,) in enumerate(data['batch_ids']):
        if batch_id in batch_ids:
            indices_to_remove.append(i)
            batch_ids.remove(batch_id)
    # reverse list of indices to enable easy removing of items of a list by consecutive pops
    indices_to_remove.reverse()
    if inplace:
        new_data = data
    else:
        new_data = {}
    new_data['input'] = np.delete(data['input'], indices_to_remove, axis=1)
    ids: list = data['batch_ids'].copy()
    for i in indices_to_remove:
        ids.pop(i)
    new_data['batch_ids'] = ids
    new_data['batch_comp'] = np.delete(data['batch_comp'], indices_to_remove, axis=0)
    if not inplace:
        new_data['target'] = {}
    for target in data['target']:
        new_data['target'][target] = np.delete(data['target'][target], indices_to_remove, axis=0)
    new_data['comps'] = np.delete(data['comps'], indices_to_remove, axis=0)

    return new_data


def get_samples_from_unprepared_data(batch_ids: set, unprepared_files: list[str], modify_batch_ids: bool = True) \
        -> list[ComputedStructureEntry]:
    if not modify_batch_ids:
        batch_ids = batch_ids.copy()
    sample = []
    for file in tqdm(unprepared_files):
        data: list[ComputedStructureEntry] = load(file)
        for entry in data:
            if entry.data['id'] in batch_ids:
                sample.append(entry)
                batch_ids.remove(entry.data['id'])
    return sample


def getfile(i: int, dir: str = 'data'):
    return os.path.join(dir, f'data_{i * 10000}_{(i + 1) * 10000}.pickle.gz')


def load(path: str):
    return pickle.load(gz.open(path, 'rb'))


def save(data, path):
    pickle.dump(data, gz.open(path, 'wb'))


def main():
    paths = ['./active_learning/sample/random_sample.pickle.gz']
    paths.extend((f'./active_learning/val/val_data_{i * 10000}_{(i + 1) * 10000}.pickle.gz' for i in range(29)))
    paths.extend((f'./active_learning/test/test_data_{i * 10000}_{(i + 1) * 10000}.pickle.gz' for i in range(29)))
    batch_ids = get_batch_ids(paths)
    data_path = 'data'
    dest_path = 'active_learning'

    for i in trange(283):
        data = load(getfile(i, data_path))
        remove_batch_ids(data, batch_ids)
        save(data, getfile(i, dest_path))


if __name__ == '__main__':
    main()
