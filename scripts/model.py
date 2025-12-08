import os
import time
from typing import Optional

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys


# ---------------------------
# WINDOWED ATTENTION MODULE
# ---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size

        # convert to channels-last
        x = x.permute(0, 2, 3, 1)  # → (B,H,W,C)

        # pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        # partition into windows
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # FlashAttention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # merge heads
        out = out.transpose(1, 2).reshape(out.size(0), ws * ws, C)

        # reverse windows
        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

        # unpad
        out = out[:, :H, :W, :]

        # back to channels-first
        out = out.permute(0, 3, 1, 2)  # → (B,C,H,W)

        return out


# ---------------------------
# DCN BLOCK WITH FIXED ATTENTION
# ---------------------------
class DCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, window=True, max_pool_kernel: int = 2, p=1, s=None):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if window:
            self.attention = WindowAttention(dim=out_channels)
        else:
            raise NotImplementedError("Only windowed attention is stable for 224x224")

        self.pool = nn.MaxPool2d(max_pool_kernel, padding=p, stride=s)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.attention(x)
        x = self.pool(x)
        return x


# ---------------------------
# FINAL DCN NETWORK
# ---------------------------
class DCNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DCNBlock(3, 32),
            DCNBlock(32, 64),
            DCNBlock(64, 96),
            DCNBlock(96, 128, max_pool_kernel=3, p=1, s=None)
        )

        # after 4 poolings: 224 → 112 → 56 → 28 → 14
        self.conv5 = nn.Conv2d(128, 32, 5)

        # adaptive pooling avoids incorrect flatten sizes
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.layers(x)
        x = F.relu(self.conv5(x))
        # x = self.gap(x).flatten(1)  # → (B, 32)
        x = x.reshape(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ARB(nn.Module):
    def __init__(self, in_channels: int, version: int = 1):
        super().__init__()
        self._version = version

        if version == 1:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=1,
                                   padding=1)
            out_channels = in_channels

        else:
            self.identity = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(1, 1),
                                      stride=2)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(3, 3),
                                   stride=2, padding=1)
            out_channels = in_channels * 2

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.att1 = WindowAttention(dim=out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.att1(out)

        if self._version == 1:
            x = out + x
        else:
            x = out + self.identity(x)
        return F.relu(self.bn2(x))


class ARBBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.arb_v1 = ARB(in_channels=in_channels, version=1)
        self.arb_v2 = ARB(in_channels=in_channels, version=2)

    def forward(self, x):
        x = self.arb_v1(x)
        x = self.arb_v2(x)
        return x


class RefocusingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(4, 4))

        self.arb1 = ARBBlock(in_channels=32)
        self.arb2 = ARBBlock(in_channels=64)
        self.arb3 = ARBBlock(in_channels=128)
        self.arb4 = ARBBlock(in_channels=256)

        # self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(in_features=8192, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        x = self.arb1(x)
        x = self.arb2(x)
        x = self.arb3(x)
        x = self.arb4(x)

        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepCascadeNetwork(nn.Module):
    def __init__(self, classification_weights_path: Optional[str] = None, positive_weights_path: Optional[str] = None,
                 negative_weights_path: Optional[str] = None):
        super(DeepCascadeNetwork, self).__init__()
        """
        Neural network described in the paper *Learning to autofocus in whole slide imaging via physics-guided deep 
        cascade networks*
        """
        self.classification_net = DCNNetwork()
        if classification_weights_path is not None:
            checkpoint = torch.load(classification_weights_path, weights_only=False)
            self.classification_net.load_state_dict(checkpoint['model_state_dict'])

        self.positive_refocus_net = RefocusingNetwork()
        if positive_weights_path is not None:
            checkpoint = torch.load(positive_weights_path, weights_only=False)
            self.positive_refocus_net.load_state_dict(checkpoint['model_state_dict'])

        self.negative_refocus_net = RefocusingNetwork()
        if negative_weights_path is not None:
            checkpoint = torch.load(negative_weights_path, weights_only=False)
            self.negative_refocus_net.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        class_logits = F.softmax(self.classification_net(x))

        classification_probability, class_predictions = torch.max(class_logits, 1)

        if class_predictions == 0:
            return (classification_probability, class_predictions), self.negative_refocus_net(x)
        else:
            return (classification_probability, class_predictions), self.positive_refocus_net(x)


if __name__ == '__main__':
    net = DeepCascadeNetwork()
    sample = torch.randn(1, 3, 224, 224)
    for i in range(5):
        start = time.time()
        output = net(torch.randn(1, 3, 224, 224))
        print(output)
        print(f'Inference time: {time.time() - start}')