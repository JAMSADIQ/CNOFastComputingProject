#!/usr/bin/env python3
"""
VTK-based Data Loader Visualization Script
Loads 3D VTK simulations, stacks channels [T,p,vx,vy,vz], and plots middle z-slice.
No Fourier features.
"""
import os
import random
import numpy as np
# patch numpy.bool for VTK compatibility
if not hasattr(np, 'bool'):
    np.bool = bool

import torch
from torch.utils.data import Dataset, DataLoader
from vtk import vtkDataSetReader
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

# Fix seeds for reproducibility
torch.manual_seed(0)
numpy_seed = 0
np.random.seed(numpy_seed)
random.seed(numpy_seed)

class WaveEquationVTKDataset(Dataset):
    def __init__(self, root_dir, sample_ids, t_in, t_out, grid_shape=(32,32,32)):
        self.root_dir = root_dir
        self.sample_ids = sample_ids
        self.t_in = t_in
        self.t_out = t_out
        self.grid_shape = grid_shape
        # validate simulation directories
        missing = [sid for sid in sample_ids if not os.path.isdir(os.path.join(root_dir, str(sid)))]
        if missing:
            raise ValueError(f"Simulation dirs not found: {missing}")
        
        # Precompute normalization constants across the training set
        # Channels: [T,p,vx,vy,vz] -> 5 channels
        # Initialize
        self.min_data = np.full(5, np.inf, dtype=np.float32)
        self.max_data = np.full(5, -np.inf, dtype=np.float32)
        self.min_model = np.full(5, np.inf, dtype=np.float32)
        self.max_model = np.full(5, -np.inf, dtype=np.float32)

        # Helper: read and stack channels for a given vtk path
        def read_and_stack(path):
            reader = vtkDataSetReader(); reader.SetFileName(path); reader.Update()
            pd = reader.GetOutput().GetPointData()
            # extract arrays
            T = vtk_to_numpy(pd.GetArray('T')).astype(np.float32)
            p = vtk_to_numpy(pd.GetArray('p')).astype(np.float32)
            v = vtk_to_numpy(pd.GetArray('v')).astype(np.float32)
            arrs = [T, p] + [v[:,i] for i in range(3)]
            stacked = np.stack(arrs, axis=0)  # [5, N]
            D,H,W = self.grid_shape
            return stacked.reshape((5, D, H, W))

        # Loop over samples to compute min/max
        for sid in self.sample_ids:
            folder = os.path.join(self.root_dir, str(sid))
            path_in  = os.path.join(folder, f"solution_{self.t_in:04d}.vtk")
            path_out = os.path.join(folder, f"solution_{self.t_out:04d}.vtk")
            arr_in  = read_and_stack(path_in)
            arr_out = read_and_stack(path_out)
            # update per-channel extrema
            for c in range(5):
                self.min_data[c] = min(self.min_data[c], arr_in[c].min())
                self.max_data[c] = max(self.max_data[c], arr_in[c].max())
                self.min_model[c] = min(self.min_model[c], arr_out[c].min())
                self.max_model[c] = max(self.max_model[c], arr_out[c].max())
        # convert to torch tensors for fast broadcast
        self.min_data = torch.from_numpy(self.min_data).view(5,1,1,1)
        self.max_data = torch.from_numpy(self.max_data).view(5,1,1,1)
        self.min_model = torch.from_numpy(self.min_model).view(5,1,1,1)
        self.max_model = torch.from_numpy(self.max_model).view(5,1,1,1)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sim = self.sample_ids[idx]
        folder = os.path.join(self.root_dir, str(sim))
        fin = os.path.join(folder, f"solution_{self.t_in:04d}.vtk")
        fout = os.path.join(folder, f"solution_{self.t_out:04d}.vtk")
        for pth in (fin, fout):
            if not os.path.isfile(pth):
                raise FileNotFoundError(f"Missing VTK file: {pth}")

        # Read and stack input
        reader = vtkDataSetReader(); reader.SetFileName(fin); reader.Update()
        pd = reader.GetOutput().GetPointData()
        T = vtk_to_numpy(pd.GetArray('T')).astype(np.float32)
        p = vtk_to_numpy(pd.GetArray('p')).astype(np.float32)
        v = vtk_to_numpy(pd.GetArray('v')).astype(np.float32)
        arrs_in = [T, p] + [v[:,i] for i in range(3)]
        inp_arr = np.stack(arrs_in, axis=0).reshape((5,)+self.grid_shape)

        # Read and stack target
        reader2 = vtkDataSetReader(); reader2.SetFileName(fout); reader2.Update()
        pd2 = reader2.GetOutput().GetPointData()
        T2 = vtk_to_numpy(pd2.GetArray('T')).astype(np.float32)
        p2 = vtk_to_numpy(pd2.GetArray('p')).astype(np.float32)
        v2 = vtk_to_numpy(pd2.GetArray('v')).astype(np.float32)
        arrs_out = [T2, p2] + [v2[:,i] for i in range(3)]
        out_arr = np.stack(arrs_out, axis=0).reshape((5,)+self.grid_shape)

        # to tensor
        inp_t = torch.from_numpy(inp_arr)
        out_t = torch.from_numpy(out_arr)
                # normalize
        inp_t = (inp_t - self.min_data) / (self.max_data - self.min_data)
        out_t = (out_t - self.min_model) / (self.max_model - self.min_model)

        return inp_t, out_t

    def get_grid(self):
        """
        Returns a (D x H x W x 3) tensor of normalized coordinates in [0,1].
        """
        D, H, W = self.grid_shape
        grid = torch.zeros((D, H, W, 3), dtype=torch.float32)
        for i in range(D):
            for j in range(H):
                for k in range(W):
                    grid[i, j, k] = torch.tensor([i/(D-1), j/(H-1), k/(W-1)])
        return grid

if __name__ == '__main__':
    # user parameters
    simulation_root = 'simulations'
    sample_ids = list(range(1,11))
    t_in = 10
    t_out = 1000
    grid_shape = (32,32,32)
    batch_size = 4
    n_plot = 5

    # variable names for channels
    var_names = ['T', 'p', 'vx', 'vy', 'vz']

    ds = WaveEquationVTKDataset(simulation_root, sample_ids, t_in, t_out, grid_shape)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    inputs, targets = next(iter(loader))  # shapes [B,5,D,H,W]
    B, C, D, H, W = inputs.shape
    mid = D // 2
    n_vis = min(n_plot, B)

    for i in range(n_vis):
        fig, axes = plt.subplots(2, C, figsize=(4*C, 8))
        fig.suptitle(f"Sample {i}, z-slice = {mid}")
        for c in range(C):
            # input
            ax = axes[0, c]
            im = ax.imshow(inputs[i,c,mid].numpy(), cmap='inferno')
            ax.set_title(f"Input: {var_names[c]}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # target
            ax = axes[1, c]
            im = ax.imshow(targets[i,c,mid].numpy(), cmap='inferno')
            ax.set_title(f"Target: {var_names[c]}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
    plt.show()

