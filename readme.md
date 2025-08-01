
# 4Deform: Neural Surface Deformation for Robust Shape Interpolation

**CVPR 2025**

[Lu Sang](https://sangluisme.github.io/), [Zehranaz Canfes](), [Dongliang Cao](https://dongliangcao.github.io/), [Riccardo Marin](https://ricma.netlify.app/), [Florian Bernard](https://scholar.google.com/citations?user=9GrQ2KYAAAAJ&hl=en), [Daniel Cremers](https://scholar.google.com/citations?user=cXQciMEAAAAJ&hl=en)

Technical University of Munich, Munich Center for Machine Learning, 
University of Bonn

[📄 PAPER](https://arxiv.org/pdf/2502.20208) [📰 Project page](https://4deform.github.io/)


![teaser](assets/teaser.png)

## 🛠️ Setup

install the package using
```
pip install -r requirements.txt
```
Please test if the `jax` successfully with `cudnn`. 

Install jax version 0.4.25 matching your CUDA version as described here. For example for CUDA 12:
```
pip install -U "jax[cuda12]"
```
We used *cuda 12.4 + cudnn v8.9.6*, Other jax versions may also work, but have not been tested.

**Trouble shooting**

- If `natsort` fail, delete the `natsort==8.4.0` in `requirements.txt`, after install the reset, run
```
pip install natsort
```
- Please make sure you successfully installed the comparable cuda version with you Jax, otherwise error ocurrs. 

## 📏 Data Preparation

We offer 2 different dataset:

- ### Shape matching data where the correspondences are obtained from method [**Unsupervised Learning of Robust Spectral Shape Matching**](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)

Please download the example [data](https://drive.google.com/file/d/1BCv3Jr1DIDxg6qiiaF4kZSj_wioEjd-e/view?usp=sharing) and extract it. We offer 2 datasets with their correspondences `Faust_r` and `shrec16_cuts`.

then run 
```
python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --corr_root <THE EXTRACT NPY FILES> --save_dir ./data/ --data_type matching
```

for example:

```
python ./datasets/preprocessing.py --data_root ./download_data/FAUST_r --corr_root ./download_data/faust_p2p --save_dir ./data/faust_r --data_type matching
```


- ### Temporal Sequence Data

    - High resolution real-world mesh sequence [**4D-DRESS**](https://eth-ait.github.io/4d-dress/).

    Please download the dataset from the website and get the folder has structure such as:

        |--- _4D-DRESS_00135_Outer_2
            |--Take19
                |--Capture
                |--Meshes_pkl
                |--Semantic
                |--SMPL
                ....

    then run 
    ```
    python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --save_dir ./data/ --data_type temporal
    ```
    for example
    ```
    python ./datasets/preprocessing.py --data_root ./_4D-DRESS_00135_Outer_2 --save_dir ./data/ --data_type dress4d
    ```

    It will create all subsequence into subfolders under dress4d, such as a `Take19` folder under `data` folder, containing
        
        |-dress4d
            |-Take19
                |--mesh
                |--ptc
                |--smpl
                |--train
                |--flax_ptc


    - Temporal Sequence Data with human and object [**BEHAVE**](https://virtualhumans.mpi-inf.mpg.de/behave/).

    Please download the dataset from the website and get the folder has structure such as:

        |--- _Date01
            |--Date01_Sub01_backpack_back
                |--t0005.000
                    |---backpack
                        |---fit01
                            |---backpack.ply
                            |---...
                        |---backpack.ply
                    |---person
                        |---fit02
                            |---person_fit.ply
                            |---...
                        |---person.ply
                    ...
                |--t0006.000
                ....

    then run 
    ```
    python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --save_dir ./data/ --data_type behave
    ```
    for example
    ```
    python ./datasets/preprocessing.py --data_root ./_Date01 --save_dir ./data/ --data_type behave
    ```

    It will create all subsequence into subfolders under dress4d, such as a `backpack_back` folder under `data` folder, containing e.g.,

        |-behave
            |-backpack_back
                |--mesh
                |--ptc
                |--smpl
                |--train
                |--flax_ptc
                |--person
                |--scene
    
    *Trouble shooting*: sometimes due to some naming issue, during processing some subfolder, error ocurrs, normally it is due to the `prefix` at [preprocessing.py](datasets/preprocessing.py) at line 84,85 doesn't fit. Please check and fix by yourself.   

## 💻 Training
To train our model, please use the following code:

```
python train.py --conf <CONFIG FILE> --savedir <SAVE PATH> --expname <NAME OF EXPERIMENT> --start_frame <INDEX OF START SHAPE> --length <LENGTH OF TOTAL SHAPE PAIRS> --reset
```

for example, train for non-temporal data `Faust_r`:

```
python train.py --conf ./conf/faust.conf --savedir ./exp --expname faust_r --start_frame 0 --length 4 --reset
```

*if you would like to continue the training instead of restart the training, delete `--reset`, the code will automatically find the last subfolder and continue from the checkpoints from that subfolder under `expname` folder.*

### Change settings in configure files

If some artifacts ocurr in the final results, you can play around the settings in the configuration files, especially change the weights in `loss` section. 


## 📺 Evaluation
to evaluation the trained model, please run

```
python eval_lipmlp.py --modeldir <SAVED TRAINING FOLDER> --steps <TIME STEP> --mc_resolution <MARCHING CUBES RESOLUTION> --external <OPTIONAL> --skip_recon <OPTIONAL>
```

- --external: optional, to evaulate on the real-world data, to get results on data such as kinect pointcloud in [**BEHAVE**](https://virtualhumans.mpi-inf.mpg.de/behave/) or [**4D-DRESS**](https://eth-ait.github.io/4d-dress/).

- --skip_recon: optional, skip the marchingcube and only evaluate velocity net, works if you only want to see results on external data. 

### Cite
```
@inproceedings{sang20254deform,
  title = {4Deform: Neural Surface Deformation for Robust Shape Interpolation},
  author = {Sang, Lu and Canfes, Zehranaz and Cao, Dongliang and Marin, Riccardo and Bernard, Florian and Cremers, Daniel},
  year = {2025},
  booktitle = {CVPR},
}

```


### Check our other work

[Implicit Neural Surface Deformation with Explicit Velocity Fields (ICLR2025)](https://github.com/Sangluisme/Implicit-surf-Deformation)

[TwoSquared: 4D Generation from 2D Image Pairs](https://sangluisme.github.io/TwoSquared/)

**We use same python envoriment as these paper, so if you encounter any problem during intall Jax with cudnn, please also check the issue in these github repos.**


### Coming

- [ ] release checkpoints
- [ ] release quantitative evaluation code
