import random

import h5py
import numpy as np
import torch
import os

from torch.utils.data import DataLoader

#from CNOModule import CNO
from CNOModule_3D import CNO
from torch.utils.data import Dataset

import scipy

from training.FourierFeatures import FourierFeatures

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

#------------------------------------------------------------------------------

#Load default parameters:
    
def default_param(network_properties):
    
    if "channel_multiplier" not in network_properties:
        network_properties["channel_multiplier"] = 32
    
    if "half_width_mult" not in network_properties:
        network_properties["half_width_mult"] = 1
    
    if "lrelu_upsampling" not in network_properties:
        network_properties["lrelu_upsampling"] = 2

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
    if "out_size" not in network_properties:
        network_properties["out_size"] = 1
    
    if "radial" not in network_properties:
        network_properties["radial_filter"] = 0
    
    if "cutoff_den" not in network_properties:
        network_properties["cutoff_den"] = 2.0001
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
        network_properties["retrain"] = 4
    
    if "kernel_size" not in network_properties:
        network_properties["kernel_size"] = 3
    
    if "activation" not in network_properties:
        #network_properties["activation"] = 'cno_lrelu'
        network_properties["activation"] = 'lrelu'
    
    return network_properties

#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, t = 5, s = 32, in_dist = True):
        
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        #Load normalization constants from the TRAINING set:
        file_data_train = "data_3d/GaussianSphereData_32x32x32_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #Default file:       
        if in_dist:
            self.file_data = "data_3d/GaussianSphereData_32x32x32_IN.h5"
        else:
            self.file_data = "data/WaveData_64x64_OUT.h5"
        
        #What time? DEFAULT : t = 5
        self.t = t
                        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0
        
        self.s = s
        if s!=32:
            self.file_data = "data/WaveData_24modes_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        #grid = torch.zeros((self.s, self.s,2))
        grid = torch.zeros((self.s, self.s, self.s, 3))
 
        for i in range(self.s):
            for j in range(self.s):
                for k in range(self.s):
                    grid[i, j, k][0] = i/(self.s - 1)
                    grid[i, j, k][1] = j/(self.s - 1)
                    grid[i, j, k][2] = k/(self.s - 1)
                
        return grid


class WaveEquation:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 32, in_dist = True):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 0
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 1, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 1, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 1, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
