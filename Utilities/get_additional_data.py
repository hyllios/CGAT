import os
import glob


def main():
    PATH = "~marques/data/material_prediction_CGAT/ternaries"
    print(glob.glob(os.path.join(PATH, '*/annotated/*.json.bz2')))


if __name__ == '__main__':
    main()
