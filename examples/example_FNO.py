#1D diffusion with FNO
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# === 1. Diffusion Data ===
def generate_diffusion_sequence(n_x=100, n_t=21, D=0.01, dx=1/100, dt=0.1):
    u = np.zeros((n_t, n_x), dtype=np.float32)
    x = np.linspace(0, 1, n_x)
    u[0] = np.exp(-100 * (x - 0.5)**2)
    for t in range(0, n_t - 1):
        u[t+1, 1:-1] = u[t, 1:-1] + D * dt / dx**2 * (u[t, :-2] - 2 * u[t, 1:-1] + u[t, 2:])
    return u

data = generate_diffusion_sequence()

# === 2. Dataset ===
class FNO1DDiffusionDataset(Dataset):
    def __init__(self, data, input_len=5, pred_len=10):
        self.input_len = input_len
        self.pred_len = pred_len
        self.data = data
        self.mean = data.mean()
        self.std = data.std() + 1e-6

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_seq = (self.data[idx:idx+self.input_len] - self.mean) / self.std
        y_seq = (self.data[idx+self.input_len:idx+self.input_len+self.pred_len] - self.mean) / self.std
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

train_dataset = FNO1DDiffusionDataset(data[:20], input_len=5, pred_len=10)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# === 3. FNO1D Module ===
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x

class FNO1D(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes=16):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):  # x: [B, C, X]
        x = self.fc0(x.permute(0, 2, 1))  # [B, X, width]
        x = x.permute(0, 2, 1)            # [B, width, X]

        x1 = self.conv1(x) + self.w1(x)
        x2 = self.conv2(x1) + self.w2(x1)
        x3 = self.conv3(x2) + self.w3(x2)

        x = x3.permute(0, 2, 1)  # [B, X, width]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)          # [B, X, T_out]
        return x.permute(0, 2, 1)  # [B, T_out, X]

# === 4. Train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO1D(in_channels=5, out_channels=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = xb.unsqueeze(1)  # [B, 1, 5, 100] → [B, 5, 100]
        xb = xb.squeeze(1)    # Make sure it's [B, 5, 100]
        xb = xb.permute(0, 1, 2)  # [B, 5, X]

        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.6f}")

# === 5. Evaluate ===
model.eval()
ds = train_dataset
with torch.no_grad():
    x_in = (data[0:5] - ds.mean) / ds.std
    x_in = torch.tensor(x_in[None, :, :], dtype=torch.float32).to(device)  # [1, 5, 100]
    pred = model(x_in)  # [1, 10, 100]
    pred = pred.cpu().squeeze().numpy()
    pred = pred * ds.std + ds.mean  # de-normalize

# === 6. Plot ===
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(10):
    ax.plot(data[5 + i], color='black', alpha=0.3, label="True" if i == 0 else "")
    ax.plot(pred[i], color='blue', alpha=0.3, label="FNO Pred" if i == 0 else "")
ax.set_title("FNO1D Multi-step Forecast: t=5 to t=15")
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

