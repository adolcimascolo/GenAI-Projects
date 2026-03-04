import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Minimal data prep and FAISS indexing utilities

def prepare_pubmedqa(input_path: str, output_path: str, limit: int = 1000):
    """Convert PubMedQA JSON into a simple JSONL with {question, answer, context}."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r') as f:
        data = json.load(f)
    cnt = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for k, v in data.items():
            if 'final_decision' in v and 'question' in v:
                record = {
                    'id': k,
                    'question': v.get('question', ''),
                    'answer': v.get('final_decision', ''),
                    'context': ' '.join(v.get('context', [])) if isinstance(v.get('context', []), list) else v.get('context', ''),
                }
                out.write(json.dumps(record) + '
')
                cnt += 1
                if cnt >= limit:
                    break
    print(f'Wrote {cnt} records to {output_path}')


def build_faiss_index(corpus_path: str, index_path: str, emb_out: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    texts = [line.strip() for line in open(corpus_path, 'r', encoding='utf-8') if line.strip()]
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(X)
    index.add(X)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(emb_out, X)
    print(f'Indexed {len(texts)} passages into {index_path} with embeddings -> {emb_out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--build-index', action='store_true')
    ap.add_argument('--input')
    ap.add_argument('--output')
    ap.add_argument('--corpus')
    ap.add_argument('--index')
    ap.add_argument('--embeddings')
    args = ap.parse_args()

    if args.prepare:
        assert args.input and args.output
        prepare_pubmedqa(args.input, args.output)
    if args.build_index:
        assert args.corpus and args.index and args.embeddings
        build_faiss_index(args.corpus, args.index, args.embeddings)
