import sys
import struct
import json

import numpy as np
import tquat
import txform
import quat
import bvh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_common import load_database, load_features, save_network

# Networks

class Extrapolator(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        super(Extrapolator, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        return self.linear2(x)

# Training procedure

if __name__ == '__main__':
    
    # Load data
    
    database = load_database('./database.bin')
    
    parents = database['bone_parents']
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    
    Ypos = database['bone_positions'].astype(np.float32)
    Yrot = database['bone_rotations'].astype(np.float32)
    Yvel = database['bone_velocities'].astype(np.float32)
    Yang = database['bone_angular_velocities'].astype(np.float32)
    
    nframes = Ypos.shape[0]
    nbones = Ypos.shape[1]
    
    # Parameters
    
    seed = 1234
    batchsize = 32
    lr = 0.001
    niter = 500000
    window = 30
    dt = 1.0 / 60.0
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    # Compute 2-component xform
    
    Ytxy = quat.to_xform_xy(Yrot)
    
    # Compute local root velocity
    
    Yrvel = quat.inv_mul_vec(Yrot[:,0], Yvel[:,0])
    Yrang = quat.inv_mul_vec(Yrot[:,0], Yang[:,0])

    # Compute means/stds

    extrapolator_mean_in = torch.as_tensor(np.hstack([
        Ypos[:,1:].mean(axis=0).ravel(),
        Ytxy[:,1:].mean(axis=0).ravel(),
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        Yrvel.mean(axis=0).ravel(),
        Yrang.mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    extrapolator_std_in = torch.as_tensor(np.hstack([
        Ypos[:,1:].std().repeat((nbones-1)*3),
        Ytxy[:,1:].std().repeat((nbones-1)*6),
        Yvel[:,1:].std().repeat((nbones-1)*3),
        Yang[:,1:].std().repeat((nbones-1)*3),
        Yrvel.std().repeat(3),
        Yrang.std().repeat(3),
    ]).astype(np.float32))

    extrapolator_mean_out = torch.as_tensor(np.hstack([
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        Yrvel.mean(axis=0).ravel(),
        Yrang.mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    extrapolator_std_out = torch.as_tensor(np.hstack([
        Yvel[:,1:].std(axis=0).ravel(),
        Yang[:,1:].std(axis=0).ravel(),
        Yrvel.std(axis=0).ravel(),
        Yrang.std(axis=0).ravel(),
    ]).astype(np.float32))
    
    Ypos = torch.as_tensor(Ypos)
    Yrot = torch.as_tensor(Yrot)
    Yvel = torch.as_tensor(Yvel)
    Yang = torch.as_tensor(Yang)
    Yrvel = torch.as_tensor(Yrvel)
    Yrang = torch.as_tensor(Yrang)
    
    # Make networks
    
    print(extrapolator_mean_in.shape, extrapolator_std_in.shape)
    print(extrapolator_mean_out.shape, extrapolator_std_out.shape)
    
    network_extrapolator = Extrapolator(len(extrapolator_mean_in), len(extrapolator_mean_out))
    
    # Function to generate test animation for evaluation
    
    example_starts = np.random.randint(0, len(Ypos) - window, size=[10])
    
    def generate_animation():
        
        with torch.no_grad():
            
            # Get slice of database for first clip
            
            for si, start in enumerate(example_starts):
                
                Ygnd_pos = Ypos[start:start+window][:,1:]
                Ygnd_rot = Yrot[start:start+window][:,1:]
                Ygnd_vel = Yvel[start:start+window][:,1:]
                Ygnd_ang = Yang[start:start+window][:,1:]
                Ygnd_rvel = Yrvel[start:start+window]
                Ygnd_rang = Yrang[start:start+window]
                    
                Ypre_pos = [Ygnd_pos[0]]
                Ypre_rot = [Ygnd_rot[0]]
                Ypre_vel = [Ygnd_vel[0]]
                Ypre_ang = [Ygnd_ang[0]]
                Ypre_rvel = [Ygnd_rvel[0]]
                Ypre_rang = [Ygnd_rang[0]]
                
                for _ in range(1, window):
                    
                    network_input = torch.cat([
                        Ypre_pos[-1].reshape([1, (nbones-1) * 3]),
                        tquat.to_xform_xy(Ypre_rot[-1]).reshape([1, (nbones-1) * 6]),
                        Ypre_vel[-1].reshape([1, (nbones-1) * 3]),
                        Ypre_ang[-1].reshape([1, (nbones-1) * 3]),
                        Ypre_rvel[-1].reshape([1, 3]),
                        Ypre_rang[-1].reshape([1, 3]),
                    ], dim=-1)
                    
                    network_output = (network_extrapolator((network_input - extrapolator_mean_in) / extrapolator_std_in) 
                        * extrapolator_std_out + extrapolator_mean_out)
                    
                    Yout_vel = network_output[:,0*(nbones-1)*3:1*(nbones-1)*3].reshape([nbones-1, 3])
                    Yout_ang = network_output[:,1*(nbones-1)*3:2*(nbones-1)*3].reshape([nbones-1, 3])
                    Yout_rvel = network_output[:,2*(nbones-1)*3+0:2*(nbones-1)*3+3].reshape([3])
                    Yout_rang = network_output[:,2*(nbones-1)*3+3:2*(nbones-1)*3+6].reshape([3])
                    
                    Ypre_pos.append(Ypre_pos[-1] + dt * Yout_vel)
                    Ypre_rot.append(tquat.mul(tquat.from_scaled_angle_axis(Yout_ang * dt), Ypre_rot[-1]))
                    Ypre_vel.append(Yout_vel)
                    Ypre_ang.append(Yout_ang)
                    Ypre_rvel.append(Yout_rvel)
                    Ypre_rang.append(Yout_rang)
                    
                Ypre_pos = torch.cat([y[None] for y in Ypre_pos], dim=0).cpu().numpy()
                Ypre_rot = torch.cat([y[None] for y in Ypre_rot], dim=0).cpu().numpy()
                Ypre_vel = torch.cat([y[None] for y in Ypre_vel], dim=0).cpu().numpy()
                Ypre_ang = torch.cat([y[None] for y in Ypre_ang], dim=0).cpu().numpy()
                Ypre_rvel = torch.cat([y[None] for y in Ypre_rvel], dim=0).cpu().numpy()
                Ypre_rang = torch.cat([y[None] for y in Ypre_rang], dim=0).cpu().numpy()
                
                # Integrate root displacement
                
                Ygnd_root_pos = Ypos[start:start+window,0].cpu().numpy()
                Ygnd_root_rot = Yrot[start:start+window,0].cpu().numpy()

                Ygnd_root_pos = quat.inv_mul_vec(Ygnd_root_rot[0:1], Ygnd_root_pos - Ygnd_root_pos[0:1])
                Ygnd_root_rot = quat.inv_mul(Ygnd_root_rot[0:1], Ygnd_root_rot)
                
                Ypre_rootpos = [Ygnd_root_pos[0]]
                Ypre_rootrot = [Ygnd_root_rot[0]]
                for i in range(1, Ygnd_pos.shape[0]):
                    Ypre_rootpos.append(Ypre_rootpos[-1] + quat.mul_vec(Ypre_rootrot[-1], Ypre_rvel[i-1]) * dt)
                    Ypre_rootrot.append(quat.mul(Ypre_rootrot[-1], quat.from_scaled_angle_axis(quat.mul_vec(Ypre_rootrot[-1], Ypre_rang[i-1]) * dt)))
                
                Ypre_rootpos = np.concatenate([p[np.newaxis] for p in Ypre_rootpos])
                Ypre_rootrot = np.concatenate([r[np.newaxis] for r in Ypre_rootrot])
                
                Ypre_pos = np.concatenate([Ypre_rootpos[:,np.newaxis], Ypre_pos], axis=1)
                Ypre_rot = np.concatenate([Ypre_rootrot[:,np.newaxis], Ypre_rot], axis=1)
                
                Ygnd_pos = np.concatenate([Ygnd_root_pos[:,np.newaxis], Ygnd_pos.cpu().numpy()], axis=1)
                Ygnd_rot = np.concatenate([Ygnd_root_rot[:,np.newaxis], Ygnd_rot.cpu().numpy()], axis=1)
                
                # Write BVH
                
                try:
                    bvh.save('extrapolator_Ygnd_%i.bvh' % si, {
                        'rotations': np.degrees(quat.to_euler(Ygnd_rot)),
                        'positions': 100.0 * Ygnd_pos,
                        'offsets': 100.0 * Ygnd_pos[0],
                        'parents': parents,
                        'names': ['joint_%i' % i for i in range(nbones)],
                        'order': 'zyx'
                    })
                    
                    bvh.save('extrapolator_Ypre_%i.bvh' % si, {
                        'rotations': np.degrees(quat.to_euler(Ypre_rot)),
                        'positions': 100.0 * Ypre_pos,
                        'offsets': 100.0 * Ypre_pos[0],
                        'parents': parents,
                        'names': ['joint_%i' % i for i in range(nbones)],
                        'order': 'zyx'
                    })
                except IOError as e:
                    print(e)
                    
    
    # Build potential batches respecting window size
    
    indices = []
    for i in range(nframes - window):
        indices.append(np.arange(i, i + window))
    indices = torch.as_tensor(np.array(indices), dtype=torch.long)
    
    # Train
    
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(
        list(network_extrapolator.parameters()), 
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
        
        Ygnd_pos = Ypos[batch][:,:,1:]
        Ygnd_rot = Yrot[batch][:,:,1:]
        Ygnd_vel = Yvel[batch][:,:,1:]
        Ygnd_ang = Yang[batch][:,:,1:]
        Ygnd_rvel = Yrvel[batch]
        Ygnd_rang = Yrang[batch]
        
        # Predict
        
        Ypre_pos = [Ygnd_pos[:,0]]
        Ypre_rot = [Ygnd_rot[:,0]]
        Ypre_vel = [Ygnd_vel[:,0]]
        Ypre_ang = [Ygnd_ang[:,0]]
        Ypre_rvel = [Ygnd_rvel[:,0]]
        Ypre_rang = [Ygnd_rang[:,0]]
        
        for _ in range(1, window):
            
            network_input = torch.cat([
                Ypre_pos[-1].reshape([batchsize, (nbones-1) * 3]),
                tquat.to_xform_xy(Ypre_rot[-1]).reshape([batchsize, (nbones-1) * 6]),
                Ypre_vel[-1].reshape([batchsize, (nbones-1) * 3]),
                Ypre_ang[-1].reshape([batchsize, (nbones-1) * 3]),
                Ypre_rvel[-1].reshape([batchsize, 3]),
                Ypre_rang[-1].reshape([batchsize, 3]),
            ], dim=-1)
            
            network_output = (network_extrapolator((network_input - extrapolator_mean_in) / extrapolator_std_in) 
                * extrapolator_std_out + extrapolator_mean_out)
            
            Yout_vel = network_output[:,0*(nbones-1)*3:1*(nbones-1)*3].reshape([batchsize, nbones-1, 3])
            Yout_ang = network_output[:,1*(nbones-1)*3:2*(nbones-1)*3].reshape([batchsize, nbones-1, 3])
            Yout_rvel = network_output[:,2*(nbones-1)*3+0:2*(nbones-1)*3+3].reshape([batchsize, 3])
            Yout_rang = network_output[:,2*(nbones-1)*3+3:2*(nbones-1)*3+6].reshape([batchsize, 3])
            
            Ypre_pos.append(Ypre_pos[-1] + dt * Yout_vel)
            Ypre_rot.append(tquat.mul(tquat.from_scaled_angle_axis(Yout_ang * dt), Ypre_rot[-1]))
            Ypre_vel.append(Yout_vel)
            Ypre_ang.append(Yout_ang)
            Ypre_rvel.append(Yout_rvel)
            Ypre_rang.append(Yout_rang)
            
        Ypre_pos = torch.cat([y[:,None] for y in Ypre_pos], dim=1)
        Ypre_rot = torch.cat([y[:,None] for y in Ypre_rot], dim=1)
        Ypre_vel = torch.cat([y[:,None] for y in Ypre_vel], dim=1)
        Ypre_ang = torch.cat([y[:,None] for y in Ypre_ang], dim=1)
        Ypre_rvel = torch.cat([y[:,None] for y in Ypre_rvel], dim=1)
        Ypre_rang = torch.cat([y[:,None] for y in Ypre_rang], dim=1)
        
        # Do FK
        
        Qgnd_rot, Qgnd_pos, Qgnd_vel, Qgnd_ang = tquat.fk_vel(
            Ygnd_rot, Ygnd_pos, Ygnd_vel, Ygnd_ang, parents)
        
        Qpre_rot, Qpre_pos, Qpre_vel, Qpre_ang = tquat.fk_vel(
            Ypre_rot, Ypre_pos, Ypre_vel, Ypre_ang, parents)
        
        # Compute losses
        
        loss_loc_pos = torch.mean(50.0 * torch.abs(Ypre_pos - Ygnd_pos))
        loss_loc_rot = torch.mean(15.0 * torch.abs(tquat.to_xform(Ypre_rot) - tquat.to_xform(Ygnd_rot)))
        loss_loc_vel = torch.mean( 5.0 * torch.abs(Ypre_vel - Ygnd_vel))
        loss_loc_ang = torch.mean( 1.5 * torch.abs(Ypre_ang - Ygnd_ang))
        
        loss_chr_pos = torch.mean( 8.0 * torch.abs(Qpre_pos - Qgnd_pos))
        loss_chr_rot = torch.mean( 5.0 * torch.abs(tquat.to_xform(Qpre_rot) - tquat.to_xform(Qgnd_rot)))
        loss_chr_vel = torch.mean( 1.2 * torch.abs(Qpre_vel - Qgnd_vel))
        loss_chr_ang = torch.mean( 0.5 * torch.abs(Qpre_ang - Qgnd_ang))
        
        loss_rvel = torch.mean( 0.5 * torch.abs(Ypre_rvel - Ygnd_rvel))
        loss_rang = torch.mean( 0.5 * torch.abs(Ypre_rang - Ygnd_rang))
        
        loss = (
            loss_loc_pos + 
            loss_loc_rot + 
            loss_loc_vel + 
            loss_loc_ang + 
            loss_chr_pos + 
            loss_chr_rot + 
            loss_chr_vel + 
            loss_chr_ang + 
            loss_rvel + 
            loss_rang)
        
        # Backprop
        
        loss.backward()

        optimizer.step()
    
        # Logging
        
        writer.add_scalar('extrapolator/loss', loss.item(), i)
        
        writer.add_scalars('extrapolator/loss_terms', {
            'loss_loc_pos': loss_loc_pos.item(),
            'loss_loc_rot': loss_loc_rot.item(),
            'loss_loc_vel': loss_loc_vel.item(),
            'loss_loc_ang': loss_loc_ang.item(),
            'loss_chr_pos': loss_chr_pos.item(),
            'loss_chr_rot': loss_chr_rot.item(),
            'loss_chr_vel': loss_chr_vel.item(),
            'loss_chr_ang': loss_chr_ang.item(),
            'loss_rvel': loss_rvel.item(),
            'loss_rang': loss_rang.item(),
        }, i)
        
        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
        
        if i % 1000 == 0:
            generate_animation()
            save_network('extrapolator.bin', [
                network_extrapolator.linear0, 
                network_extrapolator.linear1,
                network_extrapolator.linear2,
                ],
                extrapolator_mean_in,
                extrapolator_std_in,
                extrapolator_mean_out,
                extrapolator_std_out)
            
        if i % 1000 == 0:
            scheduler.step()
            