import argparse
import pandas as pd
import torch, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import VAE, vae_loss
from pathlib import Path
from rich.progress import track


def main(args):
    df = pd.read_csv(args.csv)
    X = torch.tensor(df.values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = VAE(d_in=X.shape[1], d_latent=args.latent_dim)
    optimzr = optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for (xb,) in track(dl, description=f'Epoch {epoch+1}/{args.epochs}'):
            x_hat, mu, logvar = model(xb)
            loss, _, _ = vae_loss(xb, x_hat, mu, logvar, beta=args.beta)
            optimzr.zero_grad(); loss.backward(); optimzr.step()
            tot += loss.item()*len(xb)
        print(f'Epoch {epoch+1}: loss={tot/len(ds):.4f}')

    Path('artifacts').mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'd_in': X.shape[1], 'd_latent': args.latent_dim}, 'artifacts/vae.pt')
    print('Saved artifacts/vae.pt')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--latent-dim', type=int, default=32)
    ap.add_argument('--beta', type=float, default=4.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
