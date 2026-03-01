import netCDF4
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os

# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# Config
# ============================================================

workpath = "~/mgcy/Projects/RL_forecast/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

IN_SEQ = 2
LEAD = 3

BATCH_SIZE = 1
EPOCHS_BASE = 25
EPOCHS_FT = 20

LR = 1e-3
HIDDEN = 32
PIX_FRAC = 0.10

# ============================================================
# Load Training Data
# ============================================================
files = os.listdir(workpath+"Data/Train/")
files.sort()  # Ensure consistent order

all_data = []

for fname in files:
    path = os.path.join(workpath, fname)
    print("Loading:", fname)
    ds = netCDF4.Dataset(path)
    d = np.array(ds.variables["var"])  # (C,T,H,W)
    d = np.transpose(d, (1,0,2,3))     # (T,C,H,W)
    all_data.append(d)

data = np.concatenate(all_data, axis=0)   # concat along time

T, C, H, W = data.shape
print("Combined Data shape:", data.shape)

mean = data.mean()
std = data.std()
data = (data - mean) / std

# ============================================================
# Sobolev Loss
# ============================================================

def sobolev_loss(pred, target, lambda_grad=0.1):
    error = pred - target
    mse = torch.mean(error ** 2)

    dx = (error[:, :, :, 2:] - error[:, :, :, :-2]) / 2.0
    dx = dx ** 2
    dy = (error[:, :, 2:, :] - error[:, :, :-2, :]) / 2.0
    dy = dy ** 2

    dx = dx[:, :, 1:-1, :]
    dy = dy[:, :, :, 1:-1]

    grad_loss = torch.mean(dx + dy)
    return mse + lambda_grad * grad_loss

# ============================================================
# Dataset
# ============================================================

class ForecastDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data) - LEAD

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx+2]
        y = self.data[idx+LEAD]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

loader = DataLoader(ForecastDataset(data), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# ConvLSTM
# ============================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, 3, padding=1)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        comb = torch.cat([x, h], dim=1)
        g = self.conv(comb)
        i,f,o,g = torch.chunk(g,4,dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h,c

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_ch, out_ch)
        self.cell2 = ConvLSTMCell(out_ch, out_ch)

    def forward(self, x):
        B,T,C,H,W = x.shape
        h = torch.zeros(B, self.cell1.hid_ch, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h,c = self.cell1(x[:,t], h, c)
            h,c = self.cell2(h, h, c)
        return h

# ============================================================
# UNet++ ConvLSTM
# ============================================================

class UNetPP_ConvLSTM(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.enc1 = ConvLSTMBlock(C, HIDDEN)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvLSTMBlock(HIDDEN, HIDDEN)
        self.out = nn.Conv2d(HIDDEN, C, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        p1 = p1.unsqueeze(1)
        e2 = self.enc2(p1)
        u = F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.out(u + e1)
        return y

# ============================================================
# Training (Baseline)
# ============================================================

def train(model, loader, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    for ep in range(epochs):
        tot = 0
        for x,y,_,_ in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss = sobolev_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs} | Loss={tot/len(loader):.6f}")

# ============================================================
# Build t3hat
# ============================================================

def make_t3hat(y, t1, t2):
    B,C,H,W = y.shape
    device = y.device
    mask = (torch.rand(H,W, device=device) < PIX_FRAC)
    sel = torch.rand(H,W, device=device) > 0.5
    source = torch.where(sel.unsqueeze(0).unsqueeze(0), t1, t2)
    mask = mask.unsqueeze(0).unsqueeze(0)
    yhat = torch.where(mask, source, y)
    return yhat

# ============================================================
# Channel-wise RL Agent
# ============================================================

class ChannelAgent(nn.Module):
    def __init__(self, hidden, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64, hidden*2 + out_ch),
            nn.Sigmoid()
        )

    def forward(self, s):
        g = self.net(s)
        g1 = g[:, :HIDDEN]
        g2 = g[:, HIDDEN:2*HIDDEN]
        g3 = g[:, 2*HIDDEN:]
        return g1, g2, g3

# ============================================================
# Gradient Gates
# ============================================================

def apply_gates(model, g1, g2, g3):
    # Map layer names to gate vectors
    gates = {
        "enc1.cell1.conv": g1,
        "enc1.cell2.conv": g1,
        "enc2.cell1.conv": g2,
        "enc2.cell2.conv": g2,
        "out": g3
    }
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in gates:
            gate = gates[name]
            # If gate length does not match out_channels, repeat or trim
            if gate.shape[0] != module.weight.grad.shape[0]:
                repeats = module.weight.grad.shape[0] // gate.shape[0] + 1
                gate = gate.repeat(repeats)[:module.weight.grad.shape[0]]
            gate = gate.view(-1,1,1,1)
            if module.weight.grad is not None:
                module.weight.grad *= gate
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad *= gate.view(-1)

# ============================================================
# Loss Evaluation
# ============================================================

def evaluate_loss(model, loader):
    model.eval()
    tot = 0
    with torch.no_grad():
        for x,y,_,_ in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            tot += sobolev_loss(pred,y).item()
    return tot/len(loader)

# ============================================================
# Stage 1: Baseline Training
# ============================================================

print("\nTraining Baseline...\n")
model = UNetPP_ConvLSTM(C).to(DEVICE)
train(model, loader, EPOCHS_BASE)

# ============================================================
# Stage 2: RL Fine-Tuning
# ============================================================

print("\nRL Fine-Tuning...\n")
model_ft = UNetPP_ConvLSTM(C).to(DEVICE)
model_ft.load_state_dict(model.state_dict())

optimizer = optim.Adam(model_ft.parameters(), lr=LR)
agent = ChannelAgent(HIDDEN, C).to(DEVICE)
agent_opt = optim.Adam(agent.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()

reward_history = []
g1_hist = []
g2_hist = []
g3_hist = []

prev_loss = evaluate_loss(model_ft, loader)
print("Initial FT Loss:", prev_loss)

# ============================================================
# RL Loop
# ============================================================

for ep in range(EPOCHS_FT):
    state = torch.tensor([[prev_loss]], device=DEVICE)
    g1,g2,g3 = agent(state)
    g1 = g1.squeeze(0)
    g2 = g2.squeeze(0)
    g3 = g3.squeeze(0)
    g1_hist.append(g1.detach().cpu().numpy())
    g2_hist.append(g2.detach().cpu().numpy())
    g3_hist.append(g3.detach().cpu().numpy())

    model_ft.train()
    tot = 0
    for x,y,t1,t2 in loader:
        x,y,t1,t2 = x.to(DEVICE), y.to(DEVICE), t1.to(DEVICE), t2.to(DEVICE)
        yhat = make_t3hat(y,t1,t2)
        pred = model_ft(x)
        loss = sobolev_loss(pred, yhat)
        optimizer.zero_grad()
        loss.backward()
        apply_gates(model_ft, g1, g2, g3)
        optimizer.step()
        tot += loss.item()

    curr_loss = tot/len(loader)
    reward = prev_loss - curr_loss
    reward_history.append(reward)
    logp = torch.log(g1+1e-8).sum() + torch.log(g2+1e-8).sum() + torch.log(g3+1e-8).sum()
    agent_loss = -logp * reward
    agent_opt.zero_grad()
    agent_loss.backward()
    agent_opt.step()
    prev_loss = curr_loss

    print(f"RL-FT {ep+1}/{EPOCHS_FT} | Loss={curr_loss:.6f} | Reward={reward:.6f}")

# ============================================================
# Load test data
# ============================================================

files = os.listdir(workpath+"Data/Test/")
files.sort()  # Ensure consistent order

all_data = []

for fname in files:
    path = os.path.join(workpath, fname)
    print("Loading:", fname)
    ds = netCDF4.Dataset(path)
    d = np.array(ds.variables["var"])  # (C,T,H,W)
    d = np.transpose(d, (1,0,2,3))     # (T,C,H,W)
    all_data.append(d)

data = np.concatenate(all_data, axis=0)   # concat along time

T, C, H, W = data.shape
print("Combined Data shape:", data.shape)

data = (data - mean) / std

loader = DataLoader(ForecastDataset(data), batch_size=BATCH_SIZE, shuffle=False)


print("\nTesting...\n")
model.eval()
model_ft.eval()

pred_base_list = []
pred_ft_list = []
yhat_list = []

with torch.no_grad():
    for batch_idx, (x,y,t1,t2) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        t1 = t1.to(DEVICE)
        t2 = t2.to(DEVICE)
        pred_base = model(x)
        pred_ft = model_ft(x)
        yhat = make_t3hat(y,t1,t2)

        pred_base_list.append(pred_base.cpu().numpy())
        pred_ft_list.append(pred_ft.cpu().numpy())
        yhat_list.append(yhat.cpu().numpy())


# save predictions for later analysis
np.save(os.path.join(workpath, "pred_base.npy"), np.concatenate(pred_base_list, axis=0))
np.save(os.path.join(workpath, "pred_ft.npy"), np.concatenate(pred_ft_list, axis=0))
np.save(os.path.join(workpath, "yhat.npy"), np.concatenate(yhat_list, axis=0))



# ============================================================
# Plot Heatmaps
# ============================================================

g1_arr = np.array(g1_hist)
g2_arr = np.array(g2_hist)
g3_arr = np.array(g3_hist)

fig, axes = plt.subplots(1,3, figsize=(18,6))
im1 = axes[0].imshow(g1_arr.T, aspect="auto", origin="lower", cmap="viridis")
axes[0].set_title("enc1 Channel-wise LR")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Channel")
im2 = axes[1].imshow(g2_arr.T, aspect="auto", origin="lower", cmap="viridis")
axes[1].set_title("enc2 Channel-wise LR")
axes[1].set_xlabel("Episode")
im3 = axes[2].imshow(g3_arr.T, aspect="auto", origin="lower", cmap="viridis")
axes[2].set_title("out Channel-wise LR")
axes[2].set_xlabel("Episode")
plt.colorbar(im1, ax=axes[0], fraction=0.046)
plt.colorbar(im2, ax=axes[1], fraction=0.046)
plt.colorbar(im3, ax=axes[2], fraction=0.046)
plt.suptitle("RL-Controlled Node-wise Learning Rates (Heatmaps)")
plt.tight_layout()
plt.show()

# ============================================================
# Plot Rewards
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward (Loss Reduction)")
plt.title("RL Reward Curve")
plt.grid(True)
plt.show()

print("\nAll Done.")
