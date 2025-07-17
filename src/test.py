# vtk_simulation_prediction.py

import os
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from glob import glob

# ============ Data Loading & Preprocessing ============
def read_vtk_sequence(folder):
    files = sorted(glob(os.path.join(folder, "solution_*.vtk")))
    tensor_seq = []
    for f in files:
        grid = pv.read(f)
        T = grid['T'].reshape(20, 20, 20)
        v = grid['v'].reshape(20, 20, 20, 3)
        p = grid['p'].reshape(20, 20, 20) if 'p' in grid.array_names else np.zeros_like(T)
        volume = np.stack([T, v[..., 0], v[..., 1], v[..., 2], p], axis=0)
        tensor_seq.append(volume)
    data = np.stack(tensor_seq, axis=0)
    mean = data.mean(axis=(0, 2, 3, 4), keepdims=True)
    std = data.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-5
    return ((data - mean) / std).astype(np.float32), mean, std

# will use it later
def get_obstacle_mask(obstacles_):
    mask = np.zeros((20, 20, 20), dtype=bool)
    for key, ((x1, x2), (y1, y2), (z1, z2)) in obstacles_.items():
        ix = (int(x1 * 2), int(x2 * 2) + 1)
        iy = (int(y1 * 2), int(y2 * 2) + 1)
        iz = (int(z1 * 2), int(z2 * 2) + 1)
        mask[ix[0]:ix[1], iy[0]:iy[1], iz[0]:iz[1]] = True
    return mask[None, ...]  # Add channel dim

# ============ ConvLSTM + Decoder ============
class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_lstm = nn.LSTM(input_dim * 20 * 20 * 20, hidden_dim, batch_first=True)

    def forward(self, x):
        B, T, C, H, W, D = x.size()
        x = x.view(B, T, -1)
        out, _ = self.conv_lstm(x)
        out = out.view(B, T, -1)
        return out.view(B, T, -1, H, W, D)

class SimpleDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_ch, 1)
        )

    def forward(self, x):
        return self.net(x)

class ConvLSTMForecastNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_steps):
        super().__init__()
        self.lstm = ConvLSTMBlock(input_dim, hidden_dim)
        self.decoder = SimpleDecoder(hidden_dim, input_dim)
        self.pred_steps = pred_steps

    def forward(self, x):
        lstm_out = self.lstm(x)
        last = lstm_out[:, -1]
        preds = []
        for _ in range(self.pred_steps):
            out = self.decoder(last)
            preds.append(out)
            last = out
        return torch.stack(preds, dim=1)

# ============ FNO ============
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=12):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :self.weights.shape[2], :self.weights.shape[3], :self.weights.shape[4]] = \
            torch.einsum("bcdhw,ioxyz->biodhw", x_ft, self.weights)
        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))
        return x.real

class FNO3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, 32)
        self.conv1 = SpectralConv3d(32, 32)
        self.conv2 = SpectralConv3d(32, 32)
        self.fc1 = nn.Linear(32, out_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        return x.permute(0, 4, 1, 2, 3)

# ============ Visualization ============
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_slice(pred, true, channel=0, z=10):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true[channel, :, :, z], cmap='inferno')
    plt.title("True")
    plt.subplot(1, 2, 2)
    plt.imshow(pred[channel, :, :, z], cmap='inferno')
    plt.title("Predicted")
    plt.suptitle(f"Channel {channel}, Slice z={z}")
    plt.show()

# ============ Main Trainer ============

def train_model():
    root = "/home/jsadiq/Downloads/FastComputing/simulations2/"
    sim_folders = ["8", "31", "49"]  # Add others
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTMForecastNet(input_dim=5, hidden_dim=16, pred_steps=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(10):
        total_loss = 0
        for sim in sim_folders:
            data, _, _ = read_vtk_sequence(os.path.join(root, sim))
            data = torch.tensor(data).unsqueeze(0).to(device)
            input_seq = data[:, :10]
            target_seq = data[:, 10:15]

            pred_seq = model(input_seq)
            loss = loss_fn(pred_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        train_losses.append(total_loss)
        val_losses.append(total_loss * 1.05)  # mock val

    plot_loss(train_losses, val_losses)
    with torch.no_grad():
        pred = pred_seq[0, -1].cpu().numpy()
        true = target_seq[0, -1].cpu().numpy()
        plot_slice(pred, true, channel=0)
        plot_slice(pred, true, channel=1)

if __name__ == "__main__":
    train_model()

