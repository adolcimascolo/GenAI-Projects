import argparse, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from agents import QueryPlanner, ComplianceGuard, Answer


def retrieve(query: str, k=5):
    texts = [line.strip() for line in open('data/corpus.txt', 'r', encoding='utf-8') if line.strip()]
    X = np.load('data/faiss.index.npy')
    index = faiss.read_index('data/faiss.index')
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    q = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return [texts[i] for i in I[0].tolist()]


def answer_from_snippets(question: str, snippets):
    # Placeholder: concatenate evidence and form simple answer; replace with LLM call if desired
    body = '
'.join(s[:300] for s in snippets)
    return f"Q: {question}
A: Based on the evidence below, key points are extracted.

{body}"


def main(args):
    planner = QueryPlanner()
    guard = ComplianceGuard()
    plan = planner.plan(args.question)

    evidence = []
    for sq in plan.subqueries:
        evidence.extend(retrieve(sq, k=3))

    text = answer_from_snippets(args.question, evidence[:5])
    ans = Answer(text=text, citations=evidence[:5])
    ans = guard.check(ans)

    out = {'question': args.question, 'subqueries': plan.subqueries, 'citations': ans.citations, 'answer': ans.text, 'low_confidence': ans.low_confidence}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/agentic_rag.json').write_text(json.dumps(out, indent=2))
    print('Saved reports/agentic_rag.json')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--question', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--index', required=True)
    args = ap.parse_args()
    main(args)
