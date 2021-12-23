import sys
import struct

import numpy as np
import tquat
import txform
import quat
import bvh

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_common import load_database, load_features, load_latent, save_network

# Networks

class Stepper(nn.Module):

    def __init__(self, input_size, hidden_size=512):
        super(Stepper, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return self.linear2(x)

# Training procedure

if __name__ == '__main__':
    
    # Load data
    
    database = load_database('./database.bin')
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    del database
    
    X = load_features('./features.bin')['features'].copy().astype(np.float32)
    Z = load_latent('./latent.bin')['latent'].copy().astype(np.float32)
    
    nframes = X.shape[0]
    nfeatures = X.shape[1]
    nlatent = Z.shape[1]
    
    # Parameters
    
    seed = 1234
    batchsize = 32
    lr = 0.001
    niter = 500000
    window = 20
    dt = 1.0 / 60.0
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    # Compute means/stds
    
    X_scale = X.std()
    Z_scale = Z.std()
    
    stepper_mean_out = torch.as_tensor(np.hstack([
        ((X[1:] - X[:-1]) / dt).mean(axis=0).ravel(),
        ((Z[1:] - Z[:-1]) / dt).mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    stepper_std_out = torch.as_tensor(np.hstack([
        ((X[1:] - X[:-1]) / dt).std(axis=0).ravel(),
        ((Z[1:] - Z[:-1]) / dt).std(axis=0).ravel(),
    ]).astype(np.float32))
    
    stepper_mean_in = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
        Z.mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    stepper_std_in = torch.as_tensor(np.hstack([
        X_scale.repeat(nfeatures),
        Z_scale.repeat(nlatent),
    ]).astype(np.float32))
    
    # Make PyTorch tensors
    
    X = torch.as_tensor(X)
    Z = torch.as_tensor(Z)
    
    # Make networks
    
    network_stepper = Stepper(nfeatures + nlatent)
    
    # Function to generate test predictions
    
    def generate_predictions():
        
        with torch.no_grad():
            
            # Get slice of database for first clip
            
            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])
            
            Xgnd = X[start:stop][np.newaxis]
            Zgnd = Z[start:stop][np.newaxis]
            
            # Predict, resetting every `window` frames
            
            Xtil = Xgnd.clone()
            Ztil = Zgnd.clone()
            
            for i in range(1, stop - start):
                
                if (i-1) % window == 0:
                    Xtil_prev = Xgnd[:,i-1]
                    Ztil_prev = Zgnd[:,i-1]
                else:
                    Xtil_prev = Xtil[:,i-1]
                    Ztil_prev = Ztil[:,i-1]
                
                output = (network_stepper((torch.cat([Xtil_prev, Ztil_prev], dim=-1) - 
                    stepper_mean_in) / stepper_std_in) * 
                    stepper_std_out + stepper_mean_out)
                
                Xtil[:,i] = Xtil_prev + dt * output[:,:nfeatures]
                Ztil[:,i] = Ztil_prev + dt * output[:,nfeatures:]
                
            # Write features
            
            fmin, fmax = Xgnd.cpu().numpy().min(), Xgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2*nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[0,:500,i].cpu().numpy())
                axs[i].plot(Xtil[0,:500,i].cpu().numpy())
                axs[i].set_ylim(fmin, fmax)
            plt.tight_layout()
            
            try:
                plt.savefig('stepper_X.png')
            except IOError as e:
                print(e)

            plt.close()
            
            # Write latent
            
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2*nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[0,:500,i].cpu().numpy())
                axs[i].plot(Ztil[0,:500,i].cpu().numpy())
                axs[i].set_ylim(lmin, lmax)
            plt.tight_layout()
            
            try:
                plt.savefig('stepper_Z.png')
            except IOError as e:
                print(e)

            plt.close()
    
    # Build potential batches respecting window size
    
    indices = []
    for i in range(nframes - window):
        indices.append(np.arange(i, i + window))
    indices = torch.as_tensor(np.array(indices), dtype=torch.long)
    
    # Train
    
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(
        network_stepper.parameters(), 
        lr=lr,
        amsgrad=True,
        weight_decay=0.001)
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    rolling_loss = None
    
    sys.stdout.write('\n')
    
    for i in range(niter):
    
        optimizer.zero_grad()
        
        # Extract batch
        
        batch = indices[torch.randint(0, len(indices), size=[batchsize])]
        
        Xgnd = X[batch]
        Zgnd = Z[batch]
        
        # Predict
        
        Xtil = [Xgnd[:,0]]
        Ztil = [Zgnd[:,0]]
        
        for _ in range(1, window):
            
            output = (network_stepper((torch.cat([Xtil[-1], Ztil[-1]], dim=-1) - 
                stepper_mean_in) / stepper_std_in) * 
                stepper_std_out + stepper_mean_out)
            
            Xtil.append(Xtil[-1] + dt * output[:,:nfeatures])
            Ztil.append(Ztil[-1] + dt * output[:,nfeatures:])
            
        Xtil = torch.cat([x[:,None] for x in Xtil], dim=1)
        Ztil = torch.cat([z[:,None] for z in Ztil], dim=1)
        
        # Compute velocities
        
        Xgnd_vel = (Xgnd[:,1:] - Xgnd[:,:-1]) / dt
        Zgnd_vel = (Zgnd[:,1:] - Zgnd[:,:-1]) / dt
        
        Xtil_vel = (Xtil[:,1:] - Xtil[:,:-1]) / dt
        Ztil_vel = (Ztil[:,1:] - Ztil[:,:-1]) / dt
        
        # Compute losses
        
        loss_xval = torch.mean(2.0 * torch.abs(Xgnd - Xtil))
        loss_zval = torch.mean(7.5 * torch.abs(Zgnd - Ztil))
        loss_xvel = torch.mean(0.2 * torch.abs(Xgnd_vel - Xtil_vel))
        loss_zvel = torch.mean(0.5 * torch.abs(Zgnd_vel - Ztil_vel))
        loss = loss_xval + loss_zval + loss_xvel + loss_zvel
        
        # Backprop
        
        loss.backward()

        optimizer.step()
    
        # Logging
        
        writer.add_scalar('stepper/loss', loss.item(), i)
        
        writer.add_scalars('stepper/loss_terms', {
            'xval': loss_xval.item(),
            'zval': loss_zval.item(),
            'xvel': loss_xvel.item(),
            'zvel': loss_zvel.item(),
        }, i)
        
        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
        
        if i % 1000 == 0:
            generate_predictions()
            save_network('stepper.bin', [
                network_stepper.linear0, 
                network_stepper.linear1, 
                network_stepper.linear2],
                stepper_mean_in,
                stepper_std_in,
                stepper_mean_out,
                stepper_std_out)
            
        if i % 1000 == 0:
            scheduler.step()
            