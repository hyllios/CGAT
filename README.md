# CGAT
Crystal graph attention neural networks for materials prediction

The code requires the following external packages:
* torch                     1.8.0+cu101              
* torch-cluster             1.5.9                    
* torch-geometric           1.6.3                    
* torch-scatter             2.0.6                    
* torch-sparse              0.6.9                    
* torch-spline-conv         1.2.1                    
* torchaudio                0.8.0                    
* torchvision               0.9.0+cu101              
* pytorch-lightning         1.2.4
* pymatgen                  2022.0.5
* tqdm
* numpy

newer package versions might work.



Cleaner Code will be added soon

The dataset used in the work can be found at https://archive.materialscloud.org/record/2021.128. There are some slight changes as most aflow materials denoted as possible outliers in the hull were recalculated and some systems from the materials project were updated. For the non-mixed perovskite systems the distance to the hull was recalculated with this updated dataset.
