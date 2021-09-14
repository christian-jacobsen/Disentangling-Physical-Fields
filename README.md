# Disentangling-Physical-Fields\


Variational autoencoder implementation using an architecture consisting of Dense blocks. 

Please cite (CITE). Some of the code is adapted from (CITE), see paper for more in depth explanation on architecture.

## Data

Data used in paper (cite) is too large to post here. Matlab scripts are included to generate data if desired. "Solve_Darcy.m" solves a 2D darcy flow problem and saves the data. The generative parameter distribution and factors of variation can be adjusted.

The PyTorch dataloader framework is used with "load_data_new.py" specifying the dataset. If using other data, ensure it follow this format.

## Training

Train a VAE by running "train_VAE.py". This will train a VAE with the configuration found in "config_train.py" and save to the specified location. Do not modify the name of this file. 

## Evaluation


