import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy

GRID_SHAPE = (32, 32, 32)

def load_fields(filename):
    """
    Load scalar and vector fields from a VTK file: T (temperature), vx, vy, vz.
    Returns: np.ndarray of shape [4, 32, 32, 32]
    """
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    pd = data.GetPointData()

    def safe_get_array(name, fallback=None):
        arr = pd.GetArray(name)
        if arr is not None:
            arr_np = vtk_to_numpy(arr)
            if arr_np.ndim == 1:
                return arr_np.reshape(GRID_SHAPE).astype(np.float32)
            elif arr_np.ndim == 2 and arr_np.shape[1] == 3:
                return arr_np  # vector field (not yet split)
            else:
                raise ValueError(f"Unexpected shape for {name}: {arr_np.shape}")
        return fallback if fallback is not None else np.zeros(GRID_SHAPE, dtype=np.float32)

    T = safe_get_array("T")
    v = safe_get_array("v", fallback=np.zeros((np.prod(GRID_SHAPE), 3), dtype=np.float32))
    vx, vy, vz = [v[:, i].reshape(GRID_SHAPE) for i in range(3)]

    return np.stack([T, vx, vy, vz], axis=0)  # [4, 32, 32, 32]


class VTKSequenceDataset(Dataset):
    def __init__(self, sim_dirs, root_dir="simulations", timestep_gap=10, input_seq_len=3):
        self.samples = []
        self.root_dir = root_dir
        self.gap = timestep_gap
        self.input_len = input_seq_len

        for sim in sim_dirs:
            folder = os.path.join(root_dir, str(sim))
            max_t = 100 - timestep_gap * (input_seq_len + 1)
            for t in range(10, max_t + 1, timestep_gap):
                input_files = [
                    os.path.join(folder, f"solution_{t + i * timestep_gap:04d}.vtk")
                    for i in range(input_seq_len)
                ]
                target_file = os.path.join(folder, f"solution_{t + input_seq_len * timestep_gap:04d}.vtk")

                if all(os.path.exists(f) for f in input_files + [target_file]):
                    self.samples.append((input_files, target_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_files, target_file = self.samples[idx]
        input_seq = [load_fields(f) for f in input_files]  # list of [4,32,32,32]
        target = load_fields(target_file)

        input_stack = np.stack(input_seq, axis=0)  # shape: [T, 4, 32, 32, 32]

        mean = input_stack.mean()
        std = input_stack.std() if input_stack.std() > 1e-6 else 1.0

        input_tensor = torch.from_numpy((input_stack - mean) / std).float()  # [T, 4, D, H, W]
        target_tensor = torch.from_numpy((target - mean) / std).float()      # [4, D, H, W]

        return input_tensor, target_tensor#, torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def plot_sample(input_seq, target, prediction, title_prefix=""):
    fields = ['T', 'vx', 'vy', 'vz']
    slice_idx = target.shape[1] // 2  # mid-Z slice
    fig, axs = plt.subplots(4, 4, figsize=(18, 14))  # 4 rows: each field; 4 columns: input, target, pred, error

    for i, field in enumerate(fields):
        input_slice = input_seq[-1, i, slice_idx].cpu().numpy()
        target_slice = target[i, slice_idx].cpu().numpy()
        pred_slice = prediction[i, slice_idx].detach().cpu().numpy()
        error_map = np.abs(pred_slice - target_slice)

        vmin = min(np.min(input_slice), np.min(target_slice), np.min(pred_slice))
        vmax = max(np.max(input_slice), np.max(target_slice), np.max(pred_slice))

        # Input
        im0 = axs[i, 0].imshow(input_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"{field} (Input)")
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.045)

        # Target
        im1 = axs[i, 1].imshow(target_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"{field} (Target)")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.045)

        # Prediction
        im2 = axs[i, 2].imshow(pred_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 2].set_title(f"{field} (Prediction)")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.045)

        # Error map
        im3 = axs[i, 3].imshow(error_map, cmap='inferno')
        axs[i, 3].set_title(f"{field} (|Error|)")
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.045)

        for j in range(4):
            axs[i, j].axis('off')

    # Quiver overlay on T field (use vx, vy)
    vx = prediction[1, slice_idx].cpu().numpy()
    vy = prediction[2, slice_idx].cpu().numpy()
    step = 2  # downsample for readability
    X, Y = np.meshgrid(np.arange(vx.shape[1]), np.arange(vx.shape[0]))
    axs[0, 2].quiver(X[::step, ::step], Y[::step, ::step],
                     vx[::step, ::step], vy[::step, ::step],
                     color='white', scale=50)

    plt.suptitle(f"{title_prefix} Prediction (Z-slice with error & flow)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_train_val_loss(train_losses, val_losses):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs. Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()






class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv3d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            ConvLSTMCell3D(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        self.output_conv = nn.Conv3d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, input_seq):
        # input_seq: [B, T, C, D, H, W]
        b, t, c, d, h, w = input_seq.size()
        h_states = [torch.zeros((b, self.hidden_dim, d, h, w), device=input_seq.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros((b, self.hidden_dim, d, h, w), device=input_seq.device) for _ in range(self.num_layers)]

        for t_idx in range(t):
            x = input_seq[:, t_idx]
            for l in range(self.num_layers):
                h, c = self.layers[l](x, h_states[l], c_states[l])
                h_states[l], c_states[l] = h, c
                x = h

        out = self.output_conv(h_states[-1])  # Final output
        return out  # [B, C, D, H, W]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvLSTM3D(input_dim=4, hidden_dim=16, kernel_size=3, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
#########
train_dataset = VTKSequenceDataset(sim_dirs=range(1, 10), root_dir="simulations", input_seq_len=3)
val_dataset   = VTKSequenceDataset(sim_dirs=[10], root_dir="simulations", input_seq_len=3)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)

train_losses = []
val_losses = []



for epoch in range(20):
    model.train()
    total_train_loss = 0
    for input_t, target_t, in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        input_seq = input_t.unsqueeze(1).to(device)
        target = target_t.to(device)

        pred = model(input_seq)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -- TRAIN SAMPLE PLOT --
    model.eval()
    with torch.no_grad():
        train_batch = next(iter(train_loader))
        input_seq_train, target_train = train_batch
        input_seq_train = input_seq_train.unsqueeze(1).to(device)
        target_train = target_train.to(device)
        pred_train = model(input_seq_train)
        plot_sample(
            input_seq_train[0], target_train[0], pred_train[0],
            title_prefix=f"Epoch {epoch+1} - Train",
            save_path=f"plots/epoch_{epoch+1:02d}_train.png"
        )

    # -- VALIDATION --
    total_val_loss = 0
    with torch.no_grad():
        for input_t, target_t in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            input_seq = input_t.unsqueeze(1).to(device)
            target = target_t.to(device)
            pred = model(input_seq)
            loss = loss_fn(pred, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # -- VALIDATION SAMPLE PLOT --
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        input_seq_val, target_val = val_batch
        input_seq_val = input_seq_val.unsqueeze(1).to(device)
        target_val = target_val.to(device)
        pred_val = model(input_seq_val)
        plot_sample(
            input_seq_val[0], target_val[0], pred_val[0],
            title_prefix=f"Epoch {epoch+1} - Validation",
            save_path=f"plots/epoch_{epoch+1:02d}_val.png"
        )

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


# ======== PLOT LOSSES ========
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs. Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
