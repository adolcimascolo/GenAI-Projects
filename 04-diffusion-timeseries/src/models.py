import torch, torch.nn as nn

class TinyUNet1D(nn.Module):
    def __init__(self, c=1, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, h, 3, padding=1), nn.SiLU(),
            nn.Conv1d(h, h, 3, padding=1), nn.SiLU(),
            nn.Conv1d(h, c, 3, padding=1)
        )
    def forward(self, x, t):
        return self.net(x)
