import argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--gen', required=True)
    ap.add_argument('--label-col', default='Class')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.label_col].values
    X = df.drop(columns=[args.label_col]).values.astype('float32')

    # train/test split
    n = int(0.8*len(X))
    Xtr, Ytr = X[:n], y[:n]
    Xte, Yte = X[n:], y[n:]

    # augment training with generated minority samples
    gen = np.load(args.gen)
    Xtr_aug = np.concatenate([Xtr, gen], axis=0)
    Ytr_aug = np.concatenate([Ytr, np.ones(len(gen))])

    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(Xtr, Ytr)
    base_f1 = f1_score(Yte, clf.predict(Xte))
    clf.fit(Xtr_aug, Ytr_aug)
    aug_f1 = f1_score(Yte, clf.predict(Xte))
    print({'base_f1': base_f1, 'aug_f1': aug_f1})
