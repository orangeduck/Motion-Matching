import sys
import struct

import numpy as np
import quat
import bvh

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from train_common import load_database

# Network

class MultiLinear(nn.Module):

    def __init__(self, linear_num, input_size, output_size):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty([linear_num, input_size, output_size]))
        self.bias = nn.Parameter(torch.empty([linear_num, 1, output_size]))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class PhaseEncoder(nn.Module):

    def __init__(self, input_size, hidden_size=512, phase_size=8, window_size=121):
        super(PhaseEncoder, self).__init__()
        
        self.conv0 = nn.Conv1d(input_size, hidden_size, window_size, padding='same')
        self.conv1 = nn.Conv1d(hidden_size, phase_size, window_size, padding='same')
        self.conv2 = nn.Conv1d(phase_size, hidden_size, window_size, padding='same')
        self.conv3 = nn.Conv1d(hidden_size, input_size, window_size, padding='same')
        self.multilinear = MultiLinear(phase_size, window_size, 2)
        
    def forward(self, x):
        
        N = x.shape[1]
        
        x = x.permute([0,2,1])
        
        # Convolution
        
        L = torch.tanh(self.conv1(torch.tanh(self.conv0(x))))
        
        # Compute FFT
        
        c = torch.fft.rfft(L, dim=2)
        p = torch.square(c.abs())
        f = torch.fft.rfftfreq(N, device=x.device)[None,None,:]
        
        A = 2 * torch.sqrt(torch.sum(p[:,:,1:], dim=2)) / N
        F = torch.sum((f * p)[:,:,1:], dim=2) / torch.sum(p[:,:,1:], dim=2)
        B = c[:,:,0].real / N 
        
        # Compute Phase
        
        Sxy = self.multilinear(L.permute([1, 0, 2])).permute([1, 0, 2])
        S = torch.atan2(Sxy[...,1], Sxy[...,0]) / (2 * np.pi) + 0.5
        
        # Inverse Transform
        
        T = (torch.arange(N, device=x.device) - (N // 2))
        
        Lhat = A[:,:,None] * torch.sin(2 * np.pi * (F[:,:,None] * T[None,None,:] + S[:,:,None])) + B[:,:,None]
        
        y = self.conv3(torch.tanh(self.conv2(Lhat)))
        y = y.permute([0,2,1])
        
        return y, S, A, Sxy

    
    

# Training procedure

if __name__ == '__main__':
    
    # Load data
    
    database = load_database('./database.bin')
    
    parents = database['bone_parents']
    contacts = database['contact_states']
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
    lr = 0.0001
    niter = 500000
    window = 121
    nphases = 8
    device = 'cuda'
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == 'cpu':
        torch.set_num_threads(1)
    
    # Compute world space
    
    Grot, _, Gvel, _ = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)
    
    # Compute character space
    
    Qvel = quat.inv_mul_vec(Grot[:,0:1], Gvel)
    
    # Compute means/stds
    
    encoder_mean = torch.as_tensor(Qvel.mean(axis=0).ravel(), dtype=torch.float32, device=device)
    encoder_std_in = torch.as_tensor(Qvel.std().repeat(nbones*3), dtype=torch.float32, device=device)
    encoder_std_out = torch.as_tensor(Qvel.mean(axis=0).ravel(), dtype=torch.float32, device=device)
    
    # Make PyTorch tensors
    
    Qvel = torch.as_tensor(Qvel, dtype=torch.float32, device=device)
    
    # Make network
    
    network = PhaseEncoder(len(encoder_mean), phase_size=nphases, window_size=window).to(device)
    
    # Function to generate test prediction of phases
    
    def generate_phases():
    
        with torch.no_grad():
            
            phases = []
            
            for i in range(500, 1000):
                Qgnd = Qvel[i:i+window].reshape([1, window, -1])
                phases.append(network(Qgnd)[1][0].cpu().numpy())
                
            phases = np.array(phases)
            
            fig, axs = plt.subplots(nphases, sharex=True, figsize=(12, 2*nphases))
            for i in range(nphases):
                axs[i].plot(phases[:,i])
                axs[i].set_ylim([0, 1])
            plt.tight_layout()
            
            try:
                plt.savefig('phases.png')
            except IOError as e:
                print(e)

            plt.close()
        
    # Function to generate phase database (warning: this takes some time to run...)
    
    def save_phase_database():
    
        with torch.no_grad():
            
            phases = np.empty([nframes, 2 * nphases], dtype=np.float32)
            
            for i in range(nframes):
                
                if i < window//2:
                    padding = window//2 - i
                    Qgnd = torch.cat([
                        Qvel[:1].repeat_interleave(padding, dim=0).reshape([padding, -1]),
                        Qvel[:i+window//2+1].reshape([window - padding, -1])
                    ], dim=0)[None]
                elif i > nframes - 1 - window//2:
                    padding = window//2 - (nframes - 1 - i)
                    Qgnd = torch.cat([
                        Qvel[i-window//2:].reshape([window - padding, -1]),
                        Qvel[-1:].repeat_interleave(padding, dim=0).reshape([padding, -1])
                    ], dim=0)[None]
                else:
                    Qgnd = Qvel[i-window//2:i+window//2+1].reshape([1, window, -1])
                
                _, _, A, Sxy = network(Qgnd)
                
                phases[i] = (A[...,None] * (Sxy / torch.norm(Sxy, 2, dim=-1, keepdim=True))).ravel().cpu().numpy()
                
            # Write phases
            
            with open('phases.bin', 'wb') as f:
                f.write(struct.pack('II', nframes, 2 * nphases) + phases.astype(np.float32).ravel().tobytes())
    
        
    # Build batches respecting window size
    
    indices = []
    for i in range(nframes - window):
        indices.append(np.arange(i, i + window))
    indices = torch.as_tensor(np.array(indices), dtype=torch.long, device=device)
    
    # Train
    
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(
        network.parameters(),
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
        
        Qgnd = Qvel[batch].reshape([batchsize, window, -1])
        
        # Autoencode
        
        Qprd, _, _, _ = network((Qgnd - encoder_mean) / encoder_std_in)
        Qprd = Qprd * encoder_std_out + encoder_mean
        
        # Compute losses
        
        loss = torch.mean(torch.abs(Qprd - Qgnd))
        
        # Backprop
        
        loss.backward()

        optimizer.step()
    
        # Logging
        
        writer.add_scalar('encoder/loss', loss.item(), i)

        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
            
        if i % 1000 == 0:
            generate_phases()

        if i > 0 and i % 10000 == 0:
            save_phase_database()
            
        if i % 1000 == 0:
            scheduler.step()
            