import argparse, time, json
from pathlib import Path
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import read_jsonl, write_json
from metrics import aggregate_metrics


def load_faiss(index_path, emb_path):
    index = faiss.read_index(index_path)
    X = np.load(emb_path)
    return index, X


def retrieve(index, X, embedder, query: str, texts: list, k=5):
    q = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return [texts[i] for i in I[0].tolist()]


def infer(model, tok, prompt: str, max_new_tokens=256, temperature=0.2, top_p=0.9):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    return tok.decode(out[0], skip_special_tokens=True)


def format_prompt(q, snippets):
    ctx = '
---
'.join(snippets)
    return f"You are a biomedical QA assistant. Use the provided evidence to answer.
Evidence:
{ctx}

Question: {q}
Answer with a concise, evidence-grounded statement and cite phrases from Evidence."


def main(args):
    # Lazy import to avoid heavy deps when just looking at code
    from sentence_transformers import SentenceTransformer
    texts = [line.strip() for line in open('data/corpus.txt', 'r', encoding='utf-8') if line.strip()]
    index, X = load_faiss(args.index, args.embeddings)
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')

    results = []
    for rec in read_jsonl(args.samples):
        snippets = retrieve(index, X, embedder, rec['question'], texts, k=5)
        prompt = format_prompt(rec['question'], snippets)
        t0 = time.perf_counter()
        out = infer(model, tok, prompt, max_new_tokens=args.max_new_tokens)
        latency = time.perf_counter() - t0
        results.append({'id': rec['id'], 'gold': rec['answer'], 'pred': out, 'snippets': snippets, 'latency': latency})

    metrics = aggregate_metrics(results)
    write_json({'metrics': metrics, 'results': results}, args.report)
    print('Saved report to', args.report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--report', default='reports/llm_eval.json')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    args = ap.parse_args()
    main(args)
