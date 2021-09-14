# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:20:17 2021

@author: christian jacobsen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np



class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.

    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, in_features, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.

        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                self.add_module('convT2', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, 
                       out_channels, kernel_size=kernel_size, stride=stride, 
                       padding=padding, output_padding=output_padding, bias=bias))
    return last_up


class DenseVAE(nn.Module):
    def __init__(self, data_channels, initial_features, denseblocks, growth_rate, n_latent, prior = 'std_norm', activations = nn.ReLU()):
        """
        A VAE using convolutional dense blocks and convolutional encoding layers
        """
        
        super(DenseVAE, self).__init__()
        
        self.data_channels = data_channels
        self.blocks = denseblocks
        self.K = growth_rate
        self.n_latent = n_latent
        self.act = activations
        
        enc_block_layers = self.blocks[: len(self.blocks) // 2]
        dec_block_layers = self.blocks[len(self.blocks) // 2:]
        
        
        self.enc_mean_network = nn.Sequential()
        self.dec_mean_network = nn.Sequential()
        
        self.enc_logvar = nn.Sequential()#nn.Parameter(torch.zeros(n_latent), requires_grad = True)
        self.dec_logvar = nn.Parameter(torch.zeros((65, 65)))
        
        if prior == 'scaled_gaussian':
            self.prior_logvar = torch.Tensor([np.log(1), np.log(2)]) # change the scale of prior
        elif prior == 'std_norm':
            self.prior_logvar = torch.zeros(self.n_latent) # the standard prior
            
        self.prior_logvar = self.prior_logvar.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Define the encoder
        # The first convolution which halves the image size
        self.enc_mean_network.add_module('Initial_convolution', nn.Conv2d(data_channels, initial_features, 
                                                                 kernel_size = 7, stride = 2, padding = 2, bias = False))
        
        n_features = initial_features
        
        for i, n_layers in enumerate(enc_block_layers):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.enc_mean_network.add_module('EncoderDenseBlock%d' % (i+1), block)
            n_features = n_features + n_layers * self.K
            
            enc = _Transition(in_features = n_features, out_features = n_features // 2, down = True)
            
            self.enc_mean_network.add_module('EncodeDown%d' % (i+1), enc)
            n_features = n_features // 2
            
        # now need to flatten and create layer to the latent dimensions
        flatten_dim = 16*16*n_features#16*16*n_features #64*n_features
        self.enc_mean_network.add_module('FlattenLayer', nn.Flatten())
        self.enc_mean_network.add_module('FullConn1', nn.Linear(flatten_dim, flatten_dim // 2))
        self.enc_mean_network.add_module('Act1', nn.ReLU())
        self.enc_mean_network.add_module('FullConn2', nn.Linear(flatten_dim // 2, self.n_latent)) #output of latent mean
        
        # encoder logvar
        self.enc_logvar.add_module('Initial_convolution', nn.Conv2d(data_channels, initial_features, 
                                                                 kernel_size = 7, stride = 2, padding = 2, bias = False))
        n_features = initial_features
        
        for i, n_layers in enumerate(enc_block_layers):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.enc_logvar.add_module('EncoderDenseBlock%d' % (i+1), block)
            n_features = n_features + n_layers * self.K
            
            enc = _Transition(in_features = n_features, out_features = n_features // 2, down = True)
            
            self.enc_logvar.add_module('EncodeDown%d' % (i+1), enc)
            n_features = n_features // 2
            
        flatten_dim = 16*16*n_features#16*16*n_features #64*n_features
        self.enc_logvar.add_module('FlattenLayer', nn.Flatten())
        self.enc_logvar.add_module('FullConn1', nn.Linear(flatten_dim, flatten_dim // 2))
        self.enc_logvar.add_module('Act1', nn.ReLU())
        self.enc_logvar.add_module('FullConn2', nn.Linear(flatten_dim // 2, self.n_latent)) #output of latent mean
        
        # Define the decoder taking inputs from the latent space to the output space
        self.dec_mean_network.add_module('FullConn1', nn.Linear(self.n_latent, flatten_dim // 2))
        self.dec_mean_network.add_module('Act1', nn.ReLU())
        self.dec_mean_network.add_module('FullConn2', nn.Linear(flatten_dim // 2, flatten_dim))
        self.dec_mean_network.add_module('Reshape1', Reshape((-1, n_features, 16, 16)))
        
        for i, n_layers in enumerate(dec_block_layers):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.dec_mean_network.add_module('DecoderDenseBlock%d' % (i+1), block)
            n_features += n_layers*self.K
            
            if i < len(dec_block_layers) - 1:
                dec = _Transition(in_features = n_features, out_features = n_features // 2,
                                  down = False)
                self.dec_mean_network.add_module('DecodeUp%d' % (i+1), dec)
                n_features = n_features // 2
                
        final_decode = last_decoding(n_features, data_channels, kernel_size = 4, stride = 2, padding = 1, 
                                     output_padding = 1, bias = False, drop_rate = 0)
                
        self.dec_mean_network.add_module('FinalDecode', final_decode)
        self.dec_mean_network.add_module('FinalConv', nn.Conv2d(data_channels, data_channels, 
                                                                 kernel_size = 5, stride = 1, padding = 2, bias = False))
        
    def forward(self, x):
        zmu, zlogvar = self.enc_mean_network(x), self.enc_logvar(x)
        z = self._reparameterize(zmu, zlogvar)
        
        xmu, xlogvar = self.dec_mean_network(z), self.dec_logvar
        return zmu, zlogvar, z, xmu, xlogvar
    
    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps
    
    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*(math.log(2*math.pi) + logvar + (x-mu)**2/torch.exp(logvar))
        
    def compute_kld(self, zmu, zlogvar):
        ##0.5*(zmu**2 + torch.exp(zlogvar) - 1 - zlogvar)#
        return 0.5*(zmu**2/torch.exp(self.prior_logvar) + torch.exp(zlogvar)/torch.exp(self.prior_logvar) - 1 - zlogvar + self.prior_logvar)#0.5*(2*math.log(0.25)- 0.5*torch.sum(zlogvar, 1) - 2 + 1/0.25*torch.sum(zlogvar.mul(0.5).exp_(), 1) + torch.sum((0.5-zmu)**2, 1))#
    
    def compute_loss(self, x):
        #freebits = 0
        zmu, zlogvar, z, xmu, xlogvar = self.forward(x)
        l_rec = -torch.sum(self.gaussian_log_prob(x, xmu, xlogvar), 1)
        l_reg = self.compute_kld(zmu, zlogvar)#torch.sum(F.relu(self.compute_kld(zmu, zlogvar) - freebits*math.log(2)) + freebits * math.log(2), 1)#
        return zmu, zlogvar, z, xmu, xlogvar, l_rec, l_reg
    
    def update_beta(self, beta, rec, nu, tau):
        def H(d):
            if d > 0:
                return 1.0
            else:
                return 0.0

        def f(b, d, t):
            return (1-H(d))*math.tanh(t*(b-1)) - H(d)

        return beta*math.exp(nu*f(beta, rec, tau)*rec)
    
    def compute_dis_score(self, p, z):
        # compute disentanglement score where p are true parameter samples and z are latent samples
        if p.is_cuda:
            p = p.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
        else:
            p = p.detach().numpy()
            z = z.detach().numpy()
        
        score = 0
        for i in range(z.shape[1]):
            m = np.concatenate((z[:,i].reshape((-1,1)), p), axis = 1)
            m = np.transpose(m)
            c = np.cov(m)
            score = np.max(np.abs(c[0,1:]))/np.sum(np.abs(c[0,1:])) + score
        
        return score / z.shape[1]
    
        
        
        
        
        
        
        
        
        
        
        
        



