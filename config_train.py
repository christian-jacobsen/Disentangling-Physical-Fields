"""
@author: Christian Jacobsen, University of Michigan

VAE configuration file

"""

import numpy as np
import torch



def lr_schedule_0(epoch):
    # used for VAE w/out HP network
    e0 = 5900
    if epoch < e0:
        return 0.0005
    else:
        return 0.0001
    
def lr_schedule_1(epoch):
    # used for VAE w/ HP network (more sensitive to learning rate)
    e0 = 6499
    e1 = 12500
    if epoch <= e0:
        return 0.0003
    elif epoch <= e1:
        return 0.00005
    else:
        return 0.000025


# dataset and save paths ----------------------------------------------------------------------------------------------
        
train_data_dir = 'kle2_lhs512.hdf5'   # training data directory
test_data_dir = 'kle2_mc512.hdf5'     # testing data directory

save_dir = './n2_kle2_VAEs/DenseVAE' # specify a folder where all similar models belong. 
                                     #    after training, model and configuration will be saved in a subdirectory as a .pth file
                                     
    
# architecture parameters ---------------------------------------------------------------------------------------------

n_latent = 2                # latent dimension
HP = False                 # include heirarchical prior network?

dense_blocks = [4, 6, 4]    # vector containing dense blocks and their length
growth_rate = 4             # see dense architecture for detailed explantation. dense block growth rate
data_channels = 3           # number of input channels
initial_features = 2        # see dense architecture for explanation. Features after initial convolution

if HP:
    prior = 'N/A'           # no need for prior specification w/ HP
    full_param = False      # specifies if the prior network variances are constant (False) or parameterized by NNs (True)
else:
    prior = 'std_norm'      # specify the prior:
                            #   'std_norm' = standard normal prior (isotropic gaussian)
                            #   'scaled_gaussian' = Factorized Gaussian prior centered at origin.

# training parameters --------------------------------------------------------------------------------------------------

wd = 0.                     # weight decay (Adam optimizer)
batch_size = 64             # batch size (training)
test_batch_size = 512       # not used during training, but saved for post-processing
beta0 = 0.000000001         # \beta during reconstruction-only phase

nu = 0.005
tau = 1                     # these are parameters for the beta scheduler, more details in paper

if HP:                      # specify the learning rate schedule
    lr_schedule = lr_schedule_1
    epochs = 10#14000
    rec_epochs = 5#6500
    if full_param:
        beta_p0 = beta0
        
else:
    lr_schedule = lr_schedule_0
    epochs = 10 # 6500
    rec_epochs = 5 # 4000







