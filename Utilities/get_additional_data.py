import os
import glob
import pickle
import re
import bz2
import gzip as gz
import json
from pymatgen.entries.computed_entries import ComputedEntry
from tqdm import tqdm


def get_composition(file: str):
    pattern = re.compile(r'/(A\d+B\d+C\d+)/')
    return pattern.search(file)[1]


def get_file_name(file: str):
    pattern = re.compile(r'/([\w-]*)\.json\.bz2')
    return pattern.search(file)[1]


def main():
    PATH = "/nfs/data-019/marques/data/material_prediction_CGAT/ternaries"
    files = glob.glob(os.path.join(PATH, '*', 'annotated', '*.json.bz2'))
    print(f"Found {len(files)} files.")
    new_dir = "additional_data"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for file in tqdm(files):
        dir = os.path.join(new_dir, get_composition(file))
        if not os.path.exists(dir):
            os.mkdir(dir)
        json_data = json.load(bz2.open(file, 'rb'))
        data = list(map(ComputedEntry.from_dict, json_data['entries']))
        pickle.dump(data, gz.open(os.path.join(dir), f'unprepared-{get_file_name(file)}.pickle.gz'))


if __name__ == '__main__':
    main()
