import torch, torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, d_out, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, d_out)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.LayerNorm(h), nn.LeakyReLU(0.2),
            nn.Linear(h, h), nn.LeakyReLU(0.2),
            nn.Linear(h, 1)
        )
    def forward(self, x):
        return self.net(x)
