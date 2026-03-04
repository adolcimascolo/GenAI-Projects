# WGAN-GP for Rare Event Augmentation

Implements Wasserstein GAN with Gradient Penalty to synthesize minority-class samples (e.g., rare adverse events or fraud) for class-imbalance mitigation.

## Example Data
- Kaggle Credit Card Fraud (highly imbalanced): https://www.kaggle.com/mlg-ulb/creditcardfraud

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_wgan_gp.py --csv data/creditcard.csv --epochs 50 -- minority-label 1
python src/evaluate.py --csv data/creditcard.csv --gen artifacts/gen.npy
```
