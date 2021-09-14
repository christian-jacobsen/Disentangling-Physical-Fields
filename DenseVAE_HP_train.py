# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:37:07 2021

@author: chris
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:41:10 2021

@author: chris
"""

from DenseVAE_HP_Dis import *
from DenseVAE_HP_Dis_full import *
from load_data import load_data
from load_data_new import load_data_new
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_DenseVAE_HP(path):
    config = torch.load(path)
    data_channels = 3
    initial_features = config['initial_features']
    dense_blocks = config['dense_blocks']
    growth_rate = config['growth_rate']
    n_latent = config['n_latent']
    VAE = DenseVAE_HP_Dis(data_channels, initial_features, dense_blocks, growth_rate, n_latent)
    VAE.load_state_dict(config['model_state_dict'])
    loss_reg = config['l_reg']
    loss_rec = config['l_rec']
    beta_list = config['beta_final']
    return VAE, loss_reg, loss_rec, beta_list


def DenseVAE_HP_train(train_data_dir, test_data_dir, save_dir, filename, \
                       epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                       full_param, data_channels, initial_features, dense_blocks, growth_rate, n_latent):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, train_stats = load_data_new(train_data_dir, batch_size)

    # we can load a saved state and continue training
    #VAE, loss_reg, loss_rec, beta = load_DenseVAE_HP('n2_kle2_VAEs/DenseVAE/HierarchicalPrior/DenseVAE_n2_kle2_hp_2.pth'.format(n_latent, kle))#'gauss_location_only_big_VAEs/DenseVAE_HP_0.pth')##


    if full_param:
        VAE = DenseVAE_HP_Dis_full(data_channels, initial_features, dense_blocks, growth_rate, n_latent)
    else:
        VAE = DenseVAE_HP_Dis(data_channels, initial_features, dense_blocks, growth_rate, n_latent)
    VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr_schedule(0), weight_decay = wd)
    beta = beta0
    beta_p = 0.00000001
    l_rec_list = np.zeros((epochs,))
    l_reg_list = np.zeros((epochs,))
    beta_list = np.zeros((epochs,))
    
    VAE.train()
    
    for epoch in range(epochs):

        optimizer.param_groups[0]['lr'] = lr_schedule(epoch)
               
        for n, (_, _, out_data) in enumerate(train_loader):
            out_data = out_data.to(device)
            if n==0 and epoch==0:
                if full_param:
                    _, _, _, _, _, _, _, _, _, _, l_rec, l_reg, _, _ = VAE.compute_loss(out_data)
                    l_rec_0 = torch.mean(l_rec)
                    l_reg_0 = torch.mean(l_reg)
                else:
                    _, _, _, _, _, _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(out_data)
                    l_rec_0 = torch.mean(l_rec)
                    l_reg_0 = torch.mean(l_reg)
            
            VAE.zero_grad()
            if full_param:
                _, _, _, _, _, _, _, _, _, _, l_rec, l_reg, klp, glp = VAE.compute_loss(out_data)
                
                l_rec = torch.mean(l_rec)
                #print('Lrec.. :', l_rec)
                if epoch < rec_epochs:
                    if torch.mean(l_reg) > 1e10:
                        beta = 1
                    else:
                        beta = beta0
                    
                else:
                    beta = VAE.update_beta(beta, l_rec, nu, tau)
                    beta_p = VAE.update_beta(beta_p, torch.mean(glp), nu, tau)
                    if beta > 1:
                        beta = 1
                
                
                loss = torch.mean(beta*(l_reg-klp+beta_p*klp)) + l_rec
                #print('Lreg.. :', torch.mean(l_reg))
                #print('beta.. :', beta)
                loss.backward()
                optimizer.step()
            else:
                _, _, _, _, _, _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(out_data)
                
                
                l_rec = torch.mean(l_rec)
                #print('Lrec.. :', l_rec)
                if epoch < rec_epochs:
                    loss = l_rec
                    if torch.mean(l_reg) > 1e10:
                        beta = 1
                    else:
                        beta = beta0
                else:
                    beta = VAE.update_beta(beta, l_rec, nu, tau)
                    if beta > 1:
                        beta = 1
                    
                    loss = torch.mean(beta*l_reg) + l_rec
                #print('Lreg.. :', torch.mean(l_reg))
                #print('beta.. :', beta)
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
            if full_param:
                print('beta_p = ', beta_p)
            print('l_rec = ', l_rec)
            print('l_reg = ', l_reg)
    
        for n, (true_params, _, true_data) in enumerate(train_loader):
            if n == 0:
                true_params = true_params.to(device)
                true_data = true_data.to(device)
                zmu, _, z, out_test, _, _, _, _, _, _ = VAE.forward(true_data)
                disentanglement_score = VAE.compute_dis_score(true_params, z)
                print(disentanglement_score)
        
        l_rec_list = np.insert(l_rec_list, 0, l_rec_0.cpu().detach().numpy())
        l_reg_list = np.insert(l_reg_list, 0, l_reg_0.cpu().detach().numpy())
        beta_list = np.insert(beta_list, 0, beta0)
        
        config = {'train_data_dir': train_data_dir,
                  'test_data_dir': test_data_dir,
                  'model': 'DenseVAE_HP',
                  'n_latent': n_latent,
                  'initial_features': initial_features,
                  'dense_blocks': dense_blocks,
                  'growth_rate': growth_rate,
                  'batch_size': batch_size,
                  'test_batch_size': test_batch_size,
                  'optimizer': optimizer,
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
