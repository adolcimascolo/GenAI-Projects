import argparse, numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import Generator, Discriminator
from pathlib import Path
from rich.progress import track


def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1)
    interp = alpha*real + (1-alpha)*fake
    interp.requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(outputs=d_interp, inputs=interp, grad_outputs=torch.ones_like(d_interp), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((grads.norm(2, dim=1) - 1)**2).mean()
    return gp


def main(args):
    df = pd.read_csv(args.csv)
    y = df[args.label_col].values
    X = df.drop(columns=[args.label_col]).values.astype('float32')

    # select minority class
    X_min = X[y==args.minority_label]
    ds = TensorDataset(torch.tensor(X_min))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    G = Generator(args.z_dim, X.shape[1])
    D = Discriminator(X.shape[1])
    g_opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    d_opt = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    lambda_gp = 10.0
    for epoch in range(args.epochs):
        for (xb,) in track(dl, description=f'Epoch {epoch+1}/{args.epochs}'):
            # Train D
            for _ in range(args.d_steps):
                z = torch.randn(xb.size(0), args.z_dim)
                fake = G(z).detach()
                d_real = D(xb)
                d_fake = D(fake)
                gp = gradient_penalty(D, xb, fake)
                d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp*gp
                d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Train G
            z = torch.randn(xb.size(0), args.z_dim)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        print(f'Epoch {epoch+1}: D={d_loss.item():.3f} G={g_loss.item():.3f}')

    Path('artifacts').mkdir(exist_ok=True)
    # generate a small synthetic set
    z = torch.randn(5000, args.z_dim)
    gen = G(z).detach().numpy()
    np.save('artifacts/gen.npy', gen)
    print('Saved artifacts/gen.npy')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--label-col', default='Class')
    ap.add_argument('--minority-label', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--z-dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--d-steps', type=int, default=5)
    args = ap.parse_args()
    main(args)
