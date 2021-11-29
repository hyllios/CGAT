import gzip as gz
import pickle
from pymatgen.entries.computed_entries import ComputedStructureEntry
from typing import List
import re
from tqdm import tqdm


def main():
    id_ = 0
    pattern = re.compile(r'spg(\d{1,3})')
    for i in tqdm(range(0, 2830000, 10000)):
        path = f'data_{i}_{i + 10000}.pickle.gz'
        data_list: List[ComputedStructureEntry] = pickle.load(gz.open(f'../original_data/{path}', 'rb'))
        to_remove = []
        for data in data_list:
            data.data['volume'] = data.structure.volume / data.structure.num_sites
            try:
                # use spg from Data if it is included
                data.data['id'] = f"{id_},{data.data['spg']}"
            except KeyError:
                # else try to extract it from id
                try:
                    match = pattern.search(data.data['id'])
                    spg = int(match.group(1))
                except AttributeError:
                    # calculate spg as a last resort
                    spg = data.structure.get_space_group_info()
                data.data['id'] = f"{id_},{spg}"
            # remove elements with just one element
            if len(set(data.structure.atomic_numbers)) == 1:
                to_remove.append(data)
            else:
                id_ += 1
        pickle.dump([data for data in data_list if data not in to_remove], gz.open(f'../unprepared_volume_data/{path}', 'wb'))


if __name__ == '__main__':
    main()
