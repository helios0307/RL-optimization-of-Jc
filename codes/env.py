# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 02:20:45 2025

@author: lenovo
"""

# newenv.py
# -*- coding: utf-8 -*-
"""
Optimized for reduced CPU-GPU transfers.
All heavy calculations use torch (works on CPU or GPU).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

# 坐标映射参数
X_MIN, X_MAX = -4.0, 4.0
Y_MIN, Y_MAX = -8.0, 8.0

def normalize_coord(x, x_min, x_max):
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0

def denormalize_coord(x_norm, x_min, x_max):
    return x_min + (x_norm + 1.0) * (x_max - x_min) / 2.0

def compute_C_torch(defects, bin_num=50, max_dist=18.0):
    """
    defects: tensor shape (n,2) on some device
    returns: tensor shape (bin_num,) on same device (float32)
    """
    device = defects.device
    n = defects.shape[0]
    if n < 2:
        return torch.zeros(bin_num, dtype=torch.float32, device=device)
    # pairwise distances (upper triangle)
    dists = torch.cdist(defects, defects, p=2)
    triu_idx = torch.triu_indices(n, n, offset=1, device=device)
    dists = dists[triu_idx[0], triu_idx[1]]
    # histogram using torch.histc (returns float)
    hist = torch.histc(dists, bins=bin_num, min=0.0, max=max_dist)
    denom = float(n * (n - 1) / 2.0)
    return (hist / denom).to(torch.float32)

def compute_B1_torch(defects, x_max=4.0, x_min=-4.0, y_max=8.0, y_min=-8.0, bin_num=32, max_dist=8.0):
    device = defects.device
    if defects.shape[0] == 0:
        return torch.zeros(bin_num, dtype=torch.float32, device=device)
    y = defects[:, 1]
    dists = torch.minimum(y_max - y, y - y_min).clamp(min=0.0)
    hist = torch.histc(dists, bins=bin_num, min=0.0, max=max_dist)
    return (hist / defects.shape[0]).to(torch.float32)

def compute_B2_torch(defects, x_max=4.0, x_min=-4.0, y_max=8.0, y_min=-8.0, bin_num=32, max_dist=4.0):
    device = defects.device
    if defects.shape[0] == 0:
        return torch.zeros(bin_num, dtype=torch.float32, device=device)
    x = defects[:, 0]
    dists = torch.minimum(x_max - x, x - x_min).clamp(min=0.0)
    hist = torch.histc(dists, bins=bin_num, min=0.0, max=max_dist)
    return (hist / defects.shape[0]).to(torch.float32)

def compute_S_torch(defects):
    device = defects.device
    if defects.shape[0] == 0:
        return torch.zeros(6, dtype=torch.float32, device=device)
    x = defects[:, 0]
    y = defects[:, 1]
    n = float(defects.shape[0])
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    near_count = ((y >= -5.0) & (y <= 5.0)).sum().to(torch.float32)
    return torch.tensor([n / 32.0, x_mean / 8.0, y_mean / 8.0, x_std / 8.0, y_std / 8.0, (near_count / n).clamp(0.0, 1.0)],
                        dtype=torch.float32,
                        device=device)

def compute_G_torch(defects, x_min=-4.0, x_max=4.0, y_min=-8.0, y_max=8.0, grid_size=16):
    device = defects.device
    if defects.shape[0] == 0:
        return torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)
    x = defects[:, 0]
    y = defects[:, 1]
    xs = (((x - x_min) / (x_max - x_min) * grid_size).to(torch.int64)).clamp(0, grid_size - 1)
    ys = (((y - y_min) / (y_max - y_min) * grid_size).to(torch.int64)).clamp(0, grid_size - 1)
    G = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)
    for i in range(defects.shape[0]):
        G[xs[i], ys[i]] += 1.0
    G = G / defects.shape[0]
    return G

class network(nn.Module):
    def __init__(self, c_len=50, b_len=32, s_len=6):
        super().__init__()
        self.c_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(25),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.b1_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.b2_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.s_branch = nn.Sequential(
            nn.Linear(s_len, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
        )
        self.g_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 8 + 128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, C, B1, B2, S, G):
        x_c = self.c_branch(C.unsqueeze(1))
        x_b1 = self.b1_branch(B1.unsqueeze(1))
        x_b2 = self.b2_branch(B2.unsqueeze(1))
        x_s = self.s_branch(S)
        x_g = self.g_branch(G)
        x = torch.cat([x_c, x_b1, x_b2, x_s, x_g], dim=1)
        return self.fc(x).squeeze(-1)

def load_stats(path='jc_statistics.pth', device=None):
    """加载预先计算的均值方差（如果存在），返回 dict 中包含张量（或 None）"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        stats = torch.load(path, map_location=device)
        # ensure tensors are on device
        for k in list(stats.keys()):
            stats[k] = stats[k].to(device)
        return stats
    except Exception:
        return None


def load_data_from_tensor(state_tensor, stats=None,
                          x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                          device=None):
    """
    输入:
        state_tensor: shape (30,) torch tensor on device; order: [flag,x_norm,y_norm] * 10
    返回:
        C,B1,B2,S,G 都是 torch tensors and on same device
    """
    if device is None:
        device = state_tensor.device
    # reshape to (10,3)
    x = state_tensor.reshape(10, 3)
    mask = (x[:, 0] > 0)
    if mask.sum() == 0:
        # empty
        C = torch.zeros(50, dtype=torch.float32, device=device)
        B1 = torch.zeros(32, dtype=torch.float32, device=device)
        B2 = torch.zeros(32, dtype=torch.float32, device=device)
        S = torch.zeros(6, dtype=torch.float32, device=device)
        G = torch.zeros((16, 16), dtype=torch.float32, device=device)
    else:
        coords_norm = x[mask, 1:3]
        # denormalize
        coords = torch.zeros_like(coords_norm, device=device)
        coords[:, 0] = denormalize_coord(coords_norm[:, 0], x_min, x_max)
        coords[:, 1] = denormalize_coord(coords_norm[:, 1], y_min, y_max)
        # compute features using torch
        C = compute_C_torch(coords).unsqueeze(0)  # (50,)
        # CE branch removed for simplicity (same as before)
        B1 = compute_B1_torch(coords).unsqueeze(0)
        B2 = compute_B2_torch(coords).unsqueeze(0)
        S = compute_S_torch(coords).unsqueeze(0)
        G = compute_G_torch(coords).unsqueeze(0)
    # standardize if stats provided
    if stats is not None:
        # stats keys: C_mean, C_std, B1_mean, B1_std, B2_mean, B2_std, S_mean, S_std, G_mean, G_std
        # shapes must match
        def zscore(t, mean_key, std_key):
            if mean_key in stats and std_key in stats:
                mean = stats[mean_key]
                std = stats[std_key]
                # ensure shapes compatible
                try:
                    return (t - mean) / (std + 1e-8)
                except Exception:
                    return t
            else:
                return t
        C = zscore(C, "C_mean", "C_std")
        B1 = zscore(B1, "B1_mean", "B1_std")
        B2 = zscore(B2, "B2_mean", "B2_std")
        S = zscore(S, "S_mean", "S_std")
        # G is (grid,grid) - stats maybe same shape
        G = (G - stats["G_mean"]) / (stats["G_std"] + 1e-8)
    return C.to(torch.float32), B1.to(torch.float32), B2.to(torch.float32), S.to(torch.float32), G.to(torch.float32)

class DefectEnv(gym.Env):
    """
    自定义 2D 超导缺陷环境（优化版）
    Observation: (30,) float32 : [flag,x_norm,y_norm] * 10
    Action: (30,) float32: [keep_flag, x_norm, y_norm] * 10
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, device=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.tile([0.0, -1.0, -1.0], 10).astype(np.float32),
            high=np.tile([1.0, 1.0, 1.0], 10).astype(np.float32),
            shape=(30,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.tile([0.0, -1.0, -1.0], 10).astype(np.float32),
            high=np.tile([1.0, 1.0, 1.0], 10).astype(np.float32),
            shape=(30,),
            dtype=np.float32,
        )
        self.state = None
        self.render_mode = render_mode
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Load reward model and stats once
        self.model = network().to(self.device)
        try:
            self.model.load_state_dict(torch.load('jc_predict_weights.pth', map_location=self.device))
        except Exception:
            pass
        self.model.eval()
        self.stats = load_stats('jc_statistics.pth', device=self.device)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = torch.zeros(30, dtype=torch.float32, device='cpu').numpy()
        state[0::3] = np.random.choice([0,1], size=10).astype(np.float32)
        state[1::3] = np.random.uniform(-1.0, 1.0, size=10).astype(np.float32)
        state[2::3] = np.random.uniform(-1.0, 1.0, size=10).astype(np.float32)
        self.state = state
        return self.state.copy(), {}

    def step(self, action):
        """
        action: numpy array shape (30,)
        we convert once to torch on device, compute reward on device, then return numpy obs and python reward
        """
        action = np.array(action, dtype=np.float32).flatten()
        keep_flag = (action[0::3] > 0.5).astype(np.float32)
        dx_norm = action[1::3]
        dy_norm = action[2::3]

        # update state (numpy arrays -> minimal CPU ops)
        self.state[0::3] = keep_flag
        mask = keep_flag > 0
        self.state[1::3][mask] = dx_norm[mask]
        self.state[2::3][mask] = dy_norm[mask]
        self.state[1::3] = np.clip(self.state[1::3], -1.0, 1.0)
        self.state[2::3] = np.clip(self.state[2::3], -1.0, 1.0)

        # compute reward using torch on device (single conversion)
        state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device)
        C, B1, B2, S, G = load_data_from_tensor(state_tensor, stats=self.stats, device=self.device)
        # shapes returned are batched as (1,...) so pass directly
        with torch.no_grad():
            # network expects C,B1,B2 shapes (batch, bins), S (batch,6), G (batch,2,grid,grid)
            # ensure shapes:
            C_t = C.to(self.device)
            B1_t = B1.to(self.device)
            B2_t = B2.to(self.device)
            S_t = S.to(self.device)
            G_t = G.to(self.device)
            # some adjustments in shapes for compatibility with network forward
            # our network forward expects C: (batch, bins), B1: (batch, bins), B2: (batch, bins), S: (batch,6), G: (batch,2,grid,grid)
            # compute reward
            try:
                pred = self.model(C_t, B1_t, B2_t, S_t, G_t)
                reward = float(max(-3.0, pred.squeeze(0).item()))
            except Exception:
                reward = float(-3.0)
        terminated = False
        truncated = False
        info = {}
        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.clf()
            valid_mask = self.state[0::3] > 0
            x_norm = self.state[1::3][valid_mask]
            y_norm = self.state[2::3][valid_mask]
            if len(x_norm) > 0:
                x_original = denormalize_coord(x_norm, X_MIN, X_MAX)
                y_original = denormalize_coord(y_norm, Y_MIN, Y_MAX)
                plt.scatter(x_original, y_original, c="red")
            plt.xlim(X_MIN, X_MAX)
            plt.ylim(Y_MIN, Y_MAX)
            plt.pause(0.01)

    def close(self):
        pass
