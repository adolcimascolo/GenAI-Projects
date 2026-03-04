import argparse
import pandas as pd
import torch
from models import VAE
from sklearn.manifold import TSNE


def main(args):
    df = pd.read_csv(args.csv)
    ckpt = torch.load(args.model, map_location='cpu')
    model = VAE(d_in=ckpt['d_in'], d_latent=ckpt['d_latent'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        mu, logvar = model.encode(X)
        z = mu
    print('Latent mean shape:', z.shape)
    # Optional: TSNE visualization saved to file (skipped to keep deps minimal)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    main(args)
