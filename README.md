# CGAT
Crystal graph attention neural networks for materials prediction

The code requires the following external packages:
* torch                     1.10.0+cu111              
* torch-cluster             1.5.9                    
* torch-geometric           2.0.3                    
* torch-scatter             2.0.9                    
* torch-sparse              0.6.12                    
* torch-spline-conv         1.2.1                    
* torchaudio                0.10.0                    
* torchvision               0.11.1              
* pytorch-lightning         1.5.8
* pymatgen                  2022.2.25
* tqdm
* numpy
* gpytorch 1.6.0

newer package versions might work.

pip commands tested for python 3.8:

`pip install torch==1.8.0 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html`

`pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-geometric==1.6.3`

`pip install --upgrade-strategy only-if-needed pytorch-lightning==1.5.8 torch==1.8.0+cu101`

`pip install pymatgen==2022.0.5`


The dataset used in the work can be found at https://archive.materialscloud.org/record/2021.128. There are some slight changes as most aflow materials denoted as possible outliers in the hull were recalculated and some systems from the materials project were updated. For the non-mixed perovskite systems the distance to the hull was recalculated with this updated dataset. Data for the paper "Large-scale machine-learning-assisted exploration of the whole materials space" can be found at https://archive.materialscloud.org/record/2022.126.

# Usage
The package can be installed by cloning the repository and running
```shell
pip install .
```
in the repository.

(If one wants to edit the source code installing with `pip install -e .` is advised.)

After installing one can make use of the following console scripts:
* `train-CGAT` to train a Crystal Graph Network,
* `prepare` to prepare trainings data for use with CGAT,
* `train-GP` to train Gaussian Processes.

(A full list of command line arguments can be found by running the command with `-h`.)

To test the package one can download some of the data from materials cloud, e.g., https://archive.materialscloud.org/record/file?filename=dcgat_1_000.json.bz2&record_id=1485 and convert it with the script in the README and save it.

```
import json, bz2, pickle, gzip as gz
from pymatgen.entries.computed_entries import ComputedStructureEntry

with bz2.open("dcgat_1_000.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

entries = [ComputedStructureEntry.from_dict(i) for i in data["entries"][:1000]]

print("Found " + str(len(entries)) + " entries")
print("\nEntry:\n", entries[0])
print("\nStructure:\n", entries[0].structure)
#only using the first 1000 entries to save time
pickle.dump(entries, gz.open('dcgat_1_000.pickle.gz','wb'))
```

Convert the ComputedStructureEntries to features:

`python prepare_data.py  --source-dir ../ --file dcgat_1_000.pickle.gz --target-file dcgat_1_000_features.pickle.gz --target-dir ../`

Run the training script (if necessary change the rights with chmod +x ./training_scripts/train.sh). The training script assumes 2 gpus right now. If only one is available strategy=hparams.distributed_backend needs to be removed from CGAT/train.py and --gpus set to 1.:

`./training_scripts/train.sh`

Test the model:

`python test.py --ckp tb_logs/runs/your_checkpoint.ckpt --data-path dcgat_1_000_features.pickle.gz --fea-path embeddings/matscholar-embedding.json`
