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



Cleaner Code will be added soon

The dataset used in the work can be found at https://archive.materialscloud.org/record/2021.128. There are some slight changes as most aflow materials denoted as possible outliers in the hull were recalculated and some systems from the materials project were updated. For the non-mixed perovskite systems the distance to the hull was recalculated with this updated dataset.

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
