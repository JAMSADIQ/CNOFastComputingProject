#ConvLSTM on 1D Diffusion  Just to see how this model works
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# === 1. Simulate diffusion in 1D ===
def generate_diffusion_sequence(n_x=100, n_t=21, D=0.01, dx=1/100, dt=0.1):
    u = np.zeros((n_t, n_x), dtype=np.float32)
    x = np.linspace(0, 1, n_x)
    u[0] = np.exp(-100 * (x - 0.5)**2)  # Gaussian initial condition
    for t in range(0, n_t - 1):
        u[t+1, 1:-1] = u[t, 1:-1] + D * dt / dx**2 * (u[t, :-2] - 2 * u[t, 1:-1] + u[t, 2:])
    return u

data = generate_diffusion_sequence(n_t=21)  # [21, 100]

# === 2. Dataset for multi-step prediction ===
class DiffusionMultiStepDataset(Dataset):
    def __init__(self, data, input_seq=5, pred_steps=10):
        self.data = data
        self.input_seq = input_seq
        self.pred_steps = pred_steps
        self.mean = data.mean()
        self.std = data.std() + 1e-6

    def __len__(self):
        return len(self.data) - self.input_seq - self.pred_steps + 1

    def __getitem__(self, idx):
        x_seq = self.data[idx:idx+self.input_seq]     # [T, X]
        y_seq = self.data[idx+self.input_seq:idx+self.input_seq+self.pred_steps]  # [T, X]

        # Normalize
        x_seq = (x_seq - self.mean) / self.std
        y_seq = (y_seq - self.mean) / self.std

        # Add channel and dummy height dimensions
        x_seq = torch.tensor(x_seq[:, None, :, None])  # [T, 1, X, 1]
        y_seq = torch.tensor(y_seq[:, None, :, None])  # [T, 1, X, 1]
        return x_seq, y_seq

train_dataset = DiffusionMultiStepDataset(data[:20], input_seq=5, pred_steps=10)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# === 3. ConvLSTM Model ===
class ConvLSTMCell1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=(kernel_size, 1), padding=(padding, 0))

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMForecast(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, kernel_size=5):
        super().__init__()
        self.cell = ConvLSTMCell1D(input_dim, hidden_dim, kernel_size)
        self.output = nn.Conv2d(hidden_dim, input_dim, kernel_size=(1, 1))

    def forward(self, input_seq, future_steps=10):
        B, T, C, X, _ = input_seq.size()
        h = torch.zeros((B, 16, X, 1), device=input_seq.device)
        c = torch.zeros((B, 16, X, 1), device=input_seq.device)

        # Encode history
        for t in range(T):
            h, c = self.cell(input_seq[:, t], h, c)

        # Rollout future predictions
        pred_seq = []
        x = input_seq[:, -1]
        for _ in range(future_steps):
            h, c = self.cell(x, h, c)
            out = self.output(h)
            out = torch.tanh(out) #for stability
            x = out
            pred_seq.append(out)

        return torch.stack(pred_seq, dim=1)  # [B, T_future, 1, X, 1]

# === 4. Train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMForecast().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb, future_steps=yb.shape[1])
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.6f}")

# === 5. Evaluate and Plot Rollout ===
model.eval()
with torch.no_grad():
    ds = train_dataset  # reuse same stats
    input_seq = (data[0:5] - ds.mean) / ds.std
    input_seq = torch.tensor(input_seq[:, None, :, None]).unsqueeze(0).to(device)
    pred_seq = model(input_seq, future_steps=1).squeeze().cpu().numpy()
    pred_seq = pred_seq * ds.std + ds.mean  # de-normalize

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(2):
        true_line = data[5 + i]
        pred_line = pred_seq[i]
        ax.plot(true_line, color='black', alpha=0.3, label="True" if i == 0 else "")
        ax.plot(pred_line, color='red', alpha=0.3, label="Pred" if i == 0 else "")
    ax.set_title("Multi-step Forecast (ConvLSTM) — t=5 to t=6")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

