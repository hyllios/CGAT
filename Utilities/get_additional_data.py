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
    pattern = re.compile(r'(?:/|\\)' + r'([A-Z]\d*)' + r'([A-Z]\d*)?' * 10 + r'(?:/|\\)')
    return "".join(filter(None, pattern.search(file).groups()))


def get_file_name(file: str):
    pattern = re.compile(r'([\w-]*)\.json\.bz2')
    return pattern.search(file)[1]


def main():
    PATH = "/nfs/data-019/marques/data/material_prediction_CGAT/{comp}"
    files = glob.glob(os.path.join(PATH.format(comp='binaries'), '*', 'annotated', '*.json.bz2')) + \
            glob.glob(os.path.join(PATH.format(comp='ternaries'), '*', 'annotated', '*.json.bz2'))
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


def test_get_composition():
    cases = [
        "/nfs/data-019/marques/data/material_prediction_CGAT/binaries/A2B13/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/binaries/A2B3/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/binaries/AB12/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/binaries/AB2/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/binaries/AB/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/ternaries/A2B2C5/annotated/batch-000.json.bz2",
        "/nfs/data-019/marques/data/material_prediction_CGAT/ternaries/A3B4C12/annotated/batch-000.json.bz2",
    ]
    results = [
        "A2B13",
        "A2B3",
        "AB12",
        "AB2",
        "AB",
        "A2B2C5",
        "A3B4C12"
    ]
    for case, result in zip(cases, results):
        try:
            assert get_composition(case) == result
        except AssertionError:
            print("Fails for: ", repr(case))
            print("Expected: ", repr(result))
            print("Found:", repr(get_composition(case)))


if __name__ == '__main__':
    test_get_composition()
    # main()
