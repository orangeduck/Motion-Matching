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

    def __init__(self, input_size, output_size, hidden_size=128):
        super(Extrapolator, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
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
    
    # Compute means/stds

    extrapolator_mean_in = torch.as_tensor(np.hstack([
        Ypos[:,1:].mean(axis=0).ravel(),
        quat.to_xform_xy(Yrot)[:,1:].mean(axis=0).ravel(),
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yvel[:,0]).mean(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yang[:,0]).mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    extrapolator_std_in = torch.as_tensor(np.hstack([
        Ypos[:,1:].std().repeat((nbones-1)*3),
        quat.to_xform_xy(Yrot)[:,1:].std().repeat((nbones-1)*6),
        Yvel[:,1:].std().repeat((nbones-1)*3),
        Yang[:,1:].std().repeat((nbones-1)*3),
        quat.inv_mul_vec(Yrot[:,0], Yvel[:,0]).std().repeat(3),
        quat.inv_mul_vec(Yrot[:,0], Yang[:,0]).std().repeat(3),
    ]).astype(np.float32))

    extrapolator_mean_out = torch.as_tensor(np.hstack([
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yvel[:,0]).mean(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yang[:,0]).mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    extrapolator_std_out = torch.as_tensor(np.hstack([
        Yvel[:,1:].std(axis=0).ravel(),
        Yang[:,1:].std(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yvel[:,0]).std(axis=0).ravel(),
        quat.inv_mul_vec(Yrot[:,0], Yang[:,0]).std(axis=0).ravel(),
    ]).astype(np.float32))
    
    Ypos = torch.as_tensor(Ypos)
    Yrot = torch.as_tensor(Yrot)
    Yvel = torch.as_tensor(Yvel)
    Yang = torch.as_tensor(Yang)
    
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
                
                Ygnd_pos = Ypos[start:start+window]
                Ygnd_rot = Yrot[start:start+window]
                Ygnd_vel = Yvel[start:start+window]
                Ygnd_ang = Yang[start:start+window]
                    
                Ytil_pos = [Ygnd_pos[0]]
                Ytil_rot = [Ygnd_rot[0]]
                Ytil_vel = [Ygnd_vel[0]]
                Ytil_ang = [Ygnd_ang[0]]
                
                for _ in range(1, window):
                    
                    network_input = torch.cat([
                        Ytil_pos[-1][1:].reshape([1, (nbones-1) * 3]),
                        tquat.to_xform_xy(Ytil_rot[-1][1:]).reshape([1, (nbones-1) * 6]),
                        Ytil_vel[-1][1:].reshape([1, (nbones-1) * 3]),
                        Ytil_ang[-1][1:].reshape([1, (nbones-1) * 3]),
                        tquat.inv_mul_vec(Ytil_rot[-1][0], Ytil_vel[-1][0]).reshape([1, 3]),
                        tquat.inv_mul_vec(Ytil_rot[-1][0], Ytil_ang[-1][0]).reshape([1, 3]),
                    ], dim=-1)
                    
                    network_output = (network_extrapolator((network_input - extrapolator_mean_in) / extrapolator_std_in) 
                        * extrapolator_std_out + extrapolator_mean_out)
                    
                    Yout_part_vel = network_output[:,0*(nbones-1)*3:1*(nbones-1)*3].reshape([nbones-1, 3])
                    Yout_part_ang = network_output[:,1*(nbones-1)*3:2*(nbones-1)*3].reshape([nbones-1, 3])
                    Yout_part_rvel = network_output[:,2*(nbones-1)*3+0:2*(nbones-1)*3+3].reshape([3])
                    Yout_part_rang = network_output[:,2*(nbones-1)*3+3:2*(nbones-1)*3+6].reshape([3])
                    
                    Yout_vel = torch.cat([
                        tquat.mul_vec(Ytil_rot[-1][0], Yout_part_rvel)[None],
                        Yout_part_vel], dim=0)
                    
                    Yout_ang = torch.cat([
                        tquat.mul_vec(Ytil_rot[-1][0], Yout_part_rang)[None],
                        Yout_part_ang], dim=0)
                    
                    Ytil_pos.append(Ytil_pos[-1] + dt * Yout_vel)
                    Ytil_rot.append(tquat.mul(tquat.from_scaled_angle_axis(Yout_ang * dt), Ytil_rot[-1]))
                    Ytil_vel.append(Yout_vel)
                    Ytil_ang.append(Yout_ang)
                    
                Ytil_pos = torch.cat([y[None] for y in Ytil_pos], dim=0).cpu().numpy()
                Ytil_rot = torch.cat([y[None] for y in Ytil_rot], dim=0).cpu().numpy()
                Ygnd_pos = Ygnd_pos.cpu().numpy()
                Ygnd_rot = Ygnd_rot.cpu().numpy()
                
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
                    
                    bvh.save('extrapolator_Ytil_%i.bvh' % si, {
                        'rotations': np.degrees(quat.to_euler(Ytil_rot)),
                        'positions': 100.0 * Ytil_pos,
                        'offsets': 100.0 * Ytil_pos[0],
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
        
        Ygnd_pos = Ypos[batch]
        Ygnd_rot = Yrot[batch]
        Ygnd_vel = Yvel[batch]
        Ygnd_ang = Yang[batch]
        
        # Extrapolate
        
        Ytil_pos = [Ygnd_pos[:,0]]
        Ytil_rot = [Ygnd_rot[:,0]]
        Ytil_vel = [Ygnd_vel[:,0]]
        Ytil_ang = [Ygnd_ang[:,0]]
        
        for _ in range(1, window):
            
            network_input = torch.cat([
                Ytil_pos[-1][:,1:].reshape([batchsize, (nbones-1) * 3]),
                tquat.to_xform_xy(Ytil_rot[-1][:,1:]).reshape([batchsize, (nbones-1) * 6]),
                Ytil_vel[-1][:,1:].reshape([batchsize, (nbones-1) * 3]),
                Ytil_ang[-1][:,1:].reshape([batchsize, (nbones-1) * 3]),
                tquat.inv_mul_vec(Ytil_rot[-1][:,0], Ytil_vel[-1][:,0]).reshape([batchsize, 3]),
                tquat.inv_mul_vec(Ytil_rot[-1][:,0], Ytil_ang[-1][:,0]).reshape([batchsize, 3]),
            ], dim=-1)
            
            network_output = (network_extrapolator((network_input - extrapolator_mean_in) / extrapolator_std_in) 
                * extrapolator_std_out + extrapolator_mean_out)
            
            Yout_part_vel = network_output[:,0*(nbones-1)*3:1*(nbones-1)*3].reshape([batchsize, nbones-1, 3])
            Yout_part_ang = network_output[:,1*(nbones-1)*3:2*(nbones-1)*3].reshape([batchsize, nbones-1, 3])
            Yout_part_rvel = network_output[:,2*(nbones-1)*3+0:2*(nbones-1)*3+3].reshape([batchsize, 3])
            Yout_part_rang = network_output[:,2*(nbones-1)*3+3:2*(nbones-1)*3+6].reshape([batchsize, 3])
            
            Yout_vel = torch.cat([
                tquat.mul_vec(Ytil_rot[-1][:,0], Yout_part_rvel)[:,None],
                Yout_part_vel], dim=1)
                
            Yout_ang = torch.cat([
                tquat.mul_vec(Ytil_rot[-1][:,0], Yout_part_rang)[:,None],
                Yout_part_ang], dim=1)
            
            Yout_pos = Ytil_pos[-1] + dt * Yout_vel
            Yout_rot = tquat.mul(tquat.from_scaled_angle_axis(Yout_ang * dt), Ytil_rot[-1])
            
            Ytil_pos.append(Yout_pos)
            Ytil_rot.append(Yout_rot)
            Ytil_vel.append(Yout_vel)
            Ytil_ang.append(Yout_ang)
            
        Ytil_pos = torch.cat([y[:,None] for y in Ytil_pos], dim=1)
        Ytil_rot = torch.cat([y[:,None] for y in Ytil_rot], dim=1)
        Ytil_vel = torch.cat([y[:,None] for y in Ytil_vel], dim=1)
        Ytil_ang = torch.cat([y[:,None] for y in Ytil_ang], dim=1)
        
        # Compute World Space
        
        Ggnd_rot, Ggnd_pos, Ggnd_vel, Ggnd_ang = tquat.fk_vel(
            Ygnd_rot, Ygnd_pos, Ygnd_vel, Ygnd_ang, parents)
        
        Gtil_rot, Gtil_pos, Gtil_vel, Gtil_ang = tquat.fk_vel(
            Ytil_rot, Ytil_pos, Ytil_vel, Ytil_ang, parents)
        
        # Convert to character space

        Qtil_pos = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_pos - Gtil_pos[:,:,0:1])
        Qtil_rot = tquat.inv_mul(Gtil_rot[:,:,0:1], Gtil_rot)
        Qtil_vel = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_vel)
        Qtil_ang = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_ang)
        
        Qgnd_pos = tquat.inv_mul_vec(Ggnd_rot[:,:,0:1], Ggnd_pos - Ggnd_pos[:,:,0:1])
        Qgnd_rot = tquat.inv_mul(Ggnd_rot[:,:,0:1], Ggnd_rot)
        Qgnd_vel = tquat.inv_mul_vec(Ggnd_rot[:,:,0:1], Ggnd_vel)
        Qgnd_ang = tquat.inv_mul_vec(Ggnd_rot[:,:,0:1], Ggnd_ang)
        
        # Compute losses
        
        loss_loc_pos = torch.mean(50.0 * torch.abs(Ytil_pos[:,:,1:] - Ygnd_pos[:,:,1:]))
        loss_loc_rot = torch.mean(15.0 * torch.abs(tquat.to_xform(Ytil_rot[:,:,1:]) - tquat.to_xform(Ygnd_rot[:,:,1:])))
        loss_loc_vel = torch.mean( 5.0 * torch.abs(Ytil_vel[:,:,1:] - Ygnd_vel[:,:,1:]))
        loss_loc_ang = torch.mean( 1.5 * torch.abs(Ytil_ang[:,:,1:] - Ygnd_ang[:,:,1:]))
        
        loss_chr_pos = torch.mean(15.0 * torch.abs(Qtil_pos - Qgnd_pos))
        loss_chr_rot = torch.mean( 7.5 * torch.abs(tquat.to_xform(Qtil_rot) - tquat.to_xform(Qgnd_rot)))
        loss_chr_vel = torch.mean( 1.5 * torch.abs(Qtil_vel - Qgnd_vel))
        loss_chr_ang = torch.mean( 0.8 * torch.abs(Qtil_ang - Qgnd_ang))
        
        loss = (
            loss_loc_pos + 
            loss_loc_rot + 
            loss_loc_vel + 
            loss_loc_ang + 
            loss_chr_pos + 
            loss_chr_rot + 
            loss_chr_vel + 
            loss_chr_ang)
        
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
            