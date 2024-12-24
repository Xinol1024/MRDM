# MDRL
Code for "A 3D generation framework using diffusion model and reinforcement learning to generate multi-target compounds with desired properties".

## Environment
The code need the following environment:
Package  | Version
--- | ---
Python | 3.8.19
PyTorch | 1.10.1
PyTorch Geometric | 2.0.4
RDKit | 2022.03.2
xgboost | 2.0.3

You can install the environment according to the following steps:
```python
# Create a new environment
conda create -n MDRL python=3.8 
conda activate MDRL

# Install PyTorch
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install PyTorch Geometric
conda install pyg -c pyg

# Install xgboost
conda install xgboost 

# Install RDKit
conda install -c conda-forge rdkit

# Install other tools
conda install tensorboard pyyaml easydict python-lmdb -c conda-forge
```

## Dataset
The processed data `geom_drug.tar.gz` can be downloaded from [OneDrive](https://1drv.ms/u/s!AjFub5R7uPdrk_IEIoym9vowOm1b_w?e=dHmOmi) and unzip them to `./data/geom_drug`.

## Diffusion model

### Train

The config file for training can be found in `./configs/train`. To train the model, you can run the following command:

```python
python scripts/train_drug3d.py --config <path_to_config_file> --device <device_id> --logdir <log_directory>
```
The parameters are:
- `config`: the path to the config file.
- `device`: the device to run the model.
- `logdir`: the path to save the log file.

### Sample

The config file for sampling can be found in `./configs/sample.` To sample using model, you can run the following command:
```python
python scripts/sample_drug3d.py --outdir <output_directory> --config <path_to_config_file> --device <device_id> --batch_size <batch_size>
```
The parameters are:
- `outdir`: the root directory to save the sampled molecules.
- `config`: the path to the config file.
- `device`: the device to run the model.
- `batch_size`: the batch size for sampling. 

## Reinforcement learning

### Compound-target scoring module

You can use `./RL_utils/ligand_binding_model/xgboost/xgbr_bindingdb.ipynb` to train the compound-target scoring module for the corresponding target, which is used to predict the ligand efficiency of the generated molecules in reinforcement learning.

### Reinforcement learning

First, use `./RL_utils/scoring_definition.csv` to construct the scoring target, including ligand efficiency, SA, QED, LogP, and MW. An example is given in the file.

Then, you can run the reinforcement learning using following command:
```python
python RL_utils/01_RL.py
```

