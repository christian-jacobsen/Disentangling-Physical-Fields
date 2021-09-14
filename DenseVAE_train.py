# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:41:10 2021

@author: Christian Jacobsen, University of Michigan
"""

from DenseVAE import *
from load_data_new import load_data_new
import torch
import numpy as np



def DenseVAE_train(train_data_dir, test_data_dir, save_dir, filename, \
                       epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                       data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                       prior):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # these parameters are due to the training data format of example problem ...
       
    train_loader, train_stats = load_data_new(train_data_dir, batch_size)
        
    VAE = DenseVAE(data_channels, initial_features, dense_blocks, growth_rate, n_latent, prior)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr_schedule(0), weight_decay = wd)
    beta = beta0
    
    l_rec_list = np.zeros((epochs,))
    l_reg_list = np.zeros((epochs,))
    beta_list = np.zeros((epochs,))
    
    VAE.train()
    
    for epoch in range(epochs):
        
        optimizer.param_groups[0]['lr'] = lr_schedule(epoch) #update learning rate
        
        for n, (_, _, out_data) in enumerate(train_loader):
            out_data = out_data.to(device)
            if n==0 and epoch==0: # compute initialized losses
                _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(out_data)
                l_rec_0 = torch.mean(l_rec)
                l_reg_0 = torch.mean(l_reg)
            
            
            VAE.zero_grad()
    
            _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(out_data)
            
            l_rec = torch.mean(l_rec)
            if epoch < rec_epochs:
                if torch.mean(l_reg) > 1e10:
                    beta = 1
                else:
                    beta = beta0
                loss = l_rec
            else:
                beta = VAE.update_beta(beta, l_rec, nu, tau)
                if beta > 1:
                    beta = 1
                
                loss = torch.mean(beta*l_reg) + l_rec
                
            loss.backward()
            optimizer.step()
            
            
            l_reg = torch.mean(l_reg)
            l_rec = l_rec.cpu().detach().numpy()
            l_reg = l_reg.cpu().detach().numpy()
            
            l_rec_list[epoch] = l_rec
            l_reg_list[epoch] = l_reg
            beta_list[epoch] = beta
        if epoch % 10 == 0:
            print('=======================================')
            print('Epoch = ', epoch)
            print('beta = ', beta)
            print('l_rec = ', l_rec)
            print('l_reg = ', l_reg)   
        
    for n, (true_params, _, true_data) in enumerate(train_loader):
        if n == 0:
            true_params = true_params.to(device)
            true_data = true_data.to(device)
            zmu, _, z, out_test, _ = VAE.forward(true_data)
            disentanglement_score = VAE.compute_dis_score(true_params, z)
            print(disentanglement_score)
    
    # we want to save the initialized losses also
    l_rec_list = np.insert(l_rec_list, 0, l_rec_0.cpu().detach().numpy())
    l_reg_list = np.insert(l_reg_list, 0, l_reg_0.cpu().detach().numpy())
    beta_list = np.insert(beta_list, 0, beta0)
            
    #save model
    config = {'train_data_dir': train_data_dir,
              'test_data_dir': test_data_dir,
              'model': 'DenseVAE',
              'n_latent': n_latent,
              'initial_features': initial_features,
              'dense_blocks': dense_blocks,
              'growth_rate': growth_rate,
              'batch_size': batch_size,
              'test_batch_size': test_batch_size,
              'optimizer': optimizer,
              'prior': prior,
              'beta0': beta0,
              'nu': nu,
              'tau': tau,
              'rec_epochs': rec_epochs,
              'epochs': epochs,
              'dis_score': disentanglement_score,
              'l_reg': l_reg_list,
              'l_rec': l_rec_list,
              'beta_final': beta_list,
              'model_state_dict': VAE.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(), 
              'weight_decay': wd
              }
    
    torch.save(config, save_dir + '/' + filename)
