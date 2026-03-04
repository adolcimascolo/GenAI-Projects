import argparse, os, re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

SEP = '

'

def pdfs_to_text(pdf_dir: str) -> str:
    # Placeholder: expecting pre-extracted text files; replace with pdfminer if desired
    texts = []
    for p in Path(pdf_dir).glob('**/*'):
        if p.suffix.lower() in {'.txt'}:
            texts.append(p.read_text(encoding='utf-8', errors='ignore'))
    return SEP.join(texts)


def build_index(corpus_txt: str, index_path: str, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    texts = [t for t in corpus_txt.split(SEP) if t.strip()]
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, index_path)
    np.save(index_path + '.npy', X)
    Path('data').mkdir(exist_ok=True)
    Path('data/corpus.txt').write_text('
'.join(texts), encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ingest')
    ap.add_argument('--out')
    args = ap.parse_args()
    text = pdfs_to_text(args.ingest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding='utf-8')
    build_index(text, 'data/faiss.index')
    print('Built index -> data/faiss.index and data/faiss.index.npy')
