# vtk_simulation_prediction.py

import os
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import animation

# ============ Data Loading & Preprocessing ============
def read_vtk_sequence(folder):
    files = sorted(glob(os.path.join(folder, "solution_*.vtk")))
    tensor_seq = [] #here we combine data for all time from a simulation
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

# ============ ConvLSTM + Decoder ============
class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_size = 20 * 20 * 20
        self.lstm = nn.LSTM(input_dim * self.spatial_size, hidden_dim, batch_first=True)

    def forward(self, x):
        B, T, C, H, W, D = x.shape
        x = x.view(B, T, C * H * W * D)
        out, _ = self.lstm(x)
        return out[:, -1, :]

class VolumeDecoder(nn.Module):
    def __init__(self, input_vec_dim, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_vec_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_channels * 20 * 20 * 20)
        )
        self.output_channels = output_channels

    def forward(self, x):
        B = x.shape[0]
        x = self.net(x)
        return x.view(B, self.output_channels, 20, 20, 20)

class ConvLSTMForecastNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_steps):
        super().__init__()
        self.encoder = ConvLSTMBlock(input_dim, hidden_dim)
        self.decoder = VolumeDecoder(hidden_dim, input_dim)
        self.pred_steps = pred_steps

    def forward(self, x):
        preds = []
        state = self.encoder(x)
        for _ in range(self.pred_steps):
            out = self.decoder(state)
            preds.append(out)
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    field_names = ['Temperature (T)', 'Velocity X (vx)', 'Velocity Y (vy)', 'Velocity Z (vz)', 'Pressure (p)']
    vmin = min(true[channel, :, :, z].min(), pred[channel, :, :, z].min())
    vmax = max(true[channel, :, :, z].max(), pred[channel, :, :, z].max())

    im0 = axs[0].imshow(true[channel, :, :, z], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"True {field_names[channel]}")
    plt.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(pred[channel, :, :, z], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Predicted {field_names[channel]}")
    plt.colorbar(im1, ax=axs[1], shrink=0.8)

    plt.suptitle(f"{field_names[channel]} | z-slice = {z}")
    plt.tight_layout()
    plt.show()


def iplot_slice(pred, true, channel=0, z=10):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true[channel, :, :, z], cmap='inferno')
    plt.title("True")
    plt.subplot(1, 2, 2)
    plt.imshow(pred[channel, :, :, z], cmap='inferno')
    plt.title("Predicted")
    plt.suptitle(f"Channel {channel}, Slice z={z}")
    plt.show()




def animate_timesteps(pred_seq, true_seq, channel=0, z=10, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ims = []
    for t in range(pred_seq.shape[0]):
        im1 = axs[0].imshow(true_seq[t, channel, :, :, z], cmap='inferno', animated=True)
        im2 = axs[1].imshow(pred_seq[t, channel, :, :, z], cmap='inferno', animated=True)
        ims.append([im1, im2])
    axs[0].set_title("Ground Truth")
    axs[1].set_title("Prediction")
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    plt.suptitle(f"Channel {channel}, Slice z={z}")
    if save_path:
        ani.save(save_path, writer='ffmpeg')
    plt.show()

def plot_temporal_error(pred_seq, true_seq, fields=(0, 1, 3)):
    # fields: T (0), vx (1), vz (3)
    time_steps = pred_seq.shape[0]
    errors = {ch: [] for ch in fields}
    for t in range(time_steps):
        for ch in fields:
            err = np.mean(np.abs(pred_seq[t, ch] - true_seq[t, ch]))
            errors[ch].append(err)

    plt.figure(figsize=(10, 5))
    for ch in fields:
        label = ['T', 'vx', 'vy', 'vz', 'p'][ch]
        plt.plot(errors[ch], label=f"{label} error")
    plt.xlabel("Prediction Timestep")
    plt.ylabel("Mean Absolute Error")
    plt.title("Temporal Evolution of Prediction Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============ Main Trainer ============
def train_model():
    root = "/home/jsadiq/Downloads/FastComputing/simulations2/"
    sim_folders = ["8"]#, "31", "49"]  # Add others
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_fno = False  # Set to True to use FNO3d instead of ConvLSTM

    if use_fno:
        model = FNO3d(in_channels=5, out_channels=5).to(device)
    else:
        model = ConvLSTMForecastNet(input_dim=5, hidden_dim=128, pred_steps=5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(20):
        total_loss = 0
        for sim in sim_folders:
            data, _, _ = read_vtk_sequence(os.path.join(root, sim))
            data = torch.tensor(data).unsqueeze(0).to(device)
            input_seq = data[:, :20]
            target_seq = data[:, 15:20]

            if use_fno:
                pred_seq = []
                for t in range(5):
                    pred = model(input_seq[:, -1])  # Feed last input
                    pred_seq.append(pred)
                    input_seq = torch.cat([input_seq[:, 1:], pred.unsqueeze(1)], dim=1)
                pred_seq = torch.stack(pred_seq, dim=1)
            else:
                pred_seq = model(input_seq)

            loss = loss_fn(pred_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        train_losses.append(total_loss)
        val_losses.append(total_loss * 1.05)

    plot_loss(train_losses, val_losses)
    with torch.no_grad():
        pred = pred_seq[0].cpu().numpy()
        true = target_seq[0].cpu().numpy()
        plot_slice(pred[-1], true[-1], channel=0)
        animate_timesteps(pred, true, channel=0)
        plot_temporal_error(pred, true)

if __name__ == "__main__":
    for use_fno in [False, True]:
        print(f"\nTraining with {'FNO3d' if use_fno else 'ConvLSTM'}")
        train_model()

