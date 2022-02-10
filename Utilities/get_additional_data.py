import os
import glob
import pickle
import re
import bz2
import gzip as gz
import json
from pymatgen.entries.computed_entries import ComputedStructureEntry
from CGAT.prepare_volume_data import build_dataset_prepare


def get_composition(file: str):
    pattern = re.compile(r'(A\d+B\d+C\d+)')
    return pattern.search(file)[1]


def get_file_name(file: str):
    pattern = re.compile(r'([\w-]*)\.json\.bz2')
    return pattern.search(file)[1]


def main():
    PATH = "/nfs/data-019/marques/data/material_prediction_CGAT/ternaries"
    files = glob.glob(os.path.join(PATH, '*', 'annotated', '*.json.bz2'))
    print(f"Found {len(files)} files.")
    new_dir = "additional_data"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for file in files:
        dir = os.path.join(new_dir, get_composition(file))
        if not os.path.exists(dir):
            os.mkdir(dir)
        with bz2.open(file, 'rb') as f:
            json_data = json.load(f)
        data = list(map(ComputedStructureEntry.from_dict, json_data['entries']))
        with gz.open(os.path.join(dir, f'{get_file_name(file)}.pickle.gz'), 'wb') as f:
            pickle.dump(build_dataset_prepare(data, target_property=['e_above_hull_new', 'e-form']), f)


if __name__ == '__main__':
    main()
