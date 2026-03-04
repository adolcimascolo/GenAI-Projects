# VAE for Latent-Space Compression (Tabular/Clinical)

Trains a β‑VAE to compress high-dimensional features into a smooth latent space for segmentation/anomaly detection.

## Example Data
- MIMIC-III (requires credential): https://physionet.org/content/mimiciii/1.4/
- Any CSV with 100–300+ features (normalized)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_vae.py --csv data/features.csv --epochs 20 --latent-dim 32 --beta 4.0
python src/evaluate.py --model artifacts/vae.pt --csv data/features.csv
```
