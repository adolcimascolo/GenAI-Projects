import argparse, numpy as np, torch
from models import TinyUNet1D


def sample(model, n, T, seq_len=128):
    x = torch.randn(n, 1, seq_len)
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    for t in reversed(range(T)):
        at = alphas_cumprod[t]
        noise_pred = model(x, torch.tensor([t]*n))
        x = (1/at.sqrt())*(x - (1-at)/((1-at).sqrt())*noise_pred)
        if t>0:
            x = x + betas[t].sqrt()*torch.randn_like(x)
    return x

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='artifacts/samples.npy')
    args = ap.parse_args()
    ckpt = torch.load(args.model, map_location='cpu')
    model = TinyUNet1D(); model.load_state_dict(ckpt['state_dict']); model.eval()
    x = sample(model, n=args.n, T=ckpt['steps'])
    np.save(args.out, x.numpy())
    print('Saved', args.out)
