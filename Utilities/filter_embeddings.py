from adjust_data import save, load
from tqdm import tqdm
from glob import glob
import os
import numpy as np


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
    new_data['input'] = np.delete(data['input'], indices_to_remove, axis=0)
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


def get_ids(file):
    data = load(file)
    return set([batch_id for batch_id, in data['batch_ids']])


def get_test_and_val_ids(path_to_dir):
    files = glob(os.path.join(path_to_dir, 'val', '*.pickle.gz')) + \
            glob(os.path.join(path_to_dir, 'test', '*.pickle.gz'))
    ids = set()
    for file in tqdm(files):
        ids |= get_ids(file)
    return ids


def main():
    path = 'graph_embeddings'
    target_dir = os.path.join(path, 'train')
    print('Gathering ids...')
    test_and_val_ids = get_test_and_val_ids(path)

    print('Deleting test and validation entries form training data...')
    files = glob(os.path.join(path, '*.pickle.gz'))
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    for file in tqdm(files):
        data = load(file)
        data = remove_batch_ids(data, test_and_val_ids)
        save(data, os.path.join(target_dir, os.path.basename(file)))


if __name__ == '__main__':
    main()
