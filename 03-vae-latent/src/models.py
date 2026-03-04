import torch, torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_in, d_latent=32, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, d_latent)
        self.logvar = nn.Linear(hidden, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_in))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=4.0):
    recon = ((x - x_hat)**2).mean()
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta*kld, recon, kld
