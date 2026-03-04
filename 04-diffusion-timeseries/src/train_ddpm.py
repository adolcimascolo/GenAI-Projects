import argparse, numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from rich.progress import track
from models import TinyUNet1D


def cosine_beta_schedule(T, s=0.008):
    import math
    steps = T
    xs = np.linspace(0, steps, steps+1)
    alphas_cumprod = np.cos(((xs/steps)+s)/(1+s) * math.pi/2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


def main(args):
    df = pd.read_csv(args.csv)
    X = torch.tensor(df.values, dtype=torch.float32).unsqueeze(1)  # (N, 1, T)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    betas = torch.tensor(cosine_beta_schedule(args.steps), dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    model = TinyUNet1D(c=1, h=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        tot = 0.0
        for (x,) in track(dl, description=f'Epoch {epoch+1}/{args.epochs}'):
            t = torch.randint(0, args.steps, (x.size(0),))
            at = alphas_cumprod[t].view(-1,1,1)
            noise = torch.randn_like(x)
            x_t = at.sqrt()*x + (1-at).sqrt()*noise
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(x)
        print(f'Epoch {epoch+1}: loss={tot/len(ds):.4f}')

    Path('artifacts').mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'steps': args.steps}, 'artifacts/ddpm.pt')
    print('Saved artifacts/ddpm.pt')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
