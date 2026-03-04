# 1D Diffusion (DDPM) for Synthetic Time Series

Implements a compact DDPM in PyTorch to synthesize 1D sequences (e.g., wearable signals, adherence curves).

## Example Data
- Any univariate time-window CSV (n_samples x T)
- UCI HAR / sensor-derived features (preprocessed to 1D windows)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_ddpm.py --csv data/windows.csv --steps 1000 --epochs 50
python src/evaluate.py --model artifacts/ddpm.pt --n 8 --out artifacts/samples.npy
```
