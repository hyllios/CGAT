import pickle
import gzip as gz
import os
from tqdm import tqdm
from glob import glob
from adjust_data import remove_batch_ids


def load(path):
    return pickle.load(gz.open(path))


def save(data, path):
    pickle.dump(data, gz.open(path, 'wb'))


def main():
    data_dir = 'data'
    used_data_path = os.path.join('active_learning', 'sample', 'randoms', 'random_sample_150000.pickle.gz')
    target_dir = os.path.join('new_active_learning', 'remaining')

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    print("Loading test and validation data...")
    test_val_data = load(os.path.join(data_dir, 'indices', 'test_and_val_idx.pickle.gz'))
    test_ids = set([batch_id for batch_id, in test_val_data['test_batch_ids']])
    val_ids = set([batch_id for batch_id in test_val_data['val_batch_ids']])

    print("Loading training data...")
    used_data = load(used_data_path)
    used_ids = set([batch_id for batch_id, in used_data['batch_ids']])

    ids = test_ids | val_ids | used_ids

    print("Gathering remaining data...")
    for file in tqdm(glob(os.path.join(data_dir, '*.pickle.gz'))):
        data = load(file)
        to_remove = set()

        for batch_id, in data['batch_ids']:
            if batch_id in ids:
                to_remove.add(batch_id)
                ids.remove(batch_id)

        remove_batch_ids(data, to_remove)
        save(data, os.path.join(target_dir, os.path.basename(file)))


if __name__ == '__main__':
    main()
