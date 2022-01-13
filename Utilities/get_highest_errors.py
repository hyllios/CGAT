import pandas as pd
import pickle
import gzip
from calculate_errors import get_file
from tqdm import trange
import numpy as np
from sample import get_id, search


def get_csv(i: int, path: str):
    return get_file(i, path + '/temp').replace('data', 'errors').replace('pickle.gz', 'csv')


def main():
    PATH = 'active_learning'
    UNPREPARED_PATH = 'unprepared_volume_data'

    # Start by loading the errors file
    errors = pd.DataFrame(columns=['batch_ids', 'errors'])

    print('Reading error files...')
    for i in trange(283):
        errors = errors.append(pd.read_csv(get_csv(i, PATH)), ignore_index=True)

    N = 25000
    # find the first N samples with the largest errors
    print('Sorting...')
    errors = errors.sort_values(by='errors', ascending=False, ignore_index=True).head(N)
    # convert batch_ids to a set for faster 'in' searching
    batch_ids = set(errors['batch_ids'].to_list())

    # create list for saving those samples
    new_sample = []
    print('Saving samples with highest errors...')
    for i in trange(283):
        data = pickle.load(gzip.open(get_file(i, PATH), 'rb'))
        unprepared_data = pickle.load(gzip.open(get_file(i, UNPREPARED_PATH), 'rb'))
        indices_to_remove = []
        current_batch_ids = []
        # find all batch_ids for the new sample in the current file
        for j, batch_id in enumerate(data['batch_ids']):
            batch_id = batch_id[0]
            if batch_id in batch_ids:
                indices_to_remove.append(j)
                current_batch_ids.append(get_id(batch_id))
                batch_ids.remove(batch_id)

        if len(indices_to_remove) > 0:
            # reverse order of indices for easy popping
            indices_to_remove.reverse()
            # remove those entries from data
            data['input'] = np.delete(data['input'], indices_to_remove, axis=1)
            for j in indices_to_remove:
                data['batch_ids'].pop(j)
            data['batch_comp'] = np.delete(data['batch_comp'], indices_to_remove)
            data['comps'] = np.delete(data['comps'], indices_to_remove)
            for target in data['target']:
                data['target'][target] = np.delete(data['target'][target], indices_to_remove)
            # overwrite the data with the removed entries
            pickle.dump(data, gzip.open(get_file(i, PATH), 'wb'))

            # find batch_ids in unprepared data and append to new_sample
            for batch_id in current_batch_ids:
                new_sample.append(unprepared_data.pop(search(unprepared_data, batch_id)))
    # save new sample
    pickle.dump(new_sample, gzip.open('./active_learning/sample/unprepared_sample_1.pickle.gz', 'wb'))


if __name__ == '__main__':
    main()
