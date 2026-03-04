from typing import Dict, List
import numpy as np
from sklearn.metrics import f1_score

# Toy metrics for demo purposes

def citation_overlap(answer: str, snippets: List[str]) -> float:
    """Compute fraction of unique tokens in answer that appear in any snippet (proxy)."""
    toks = set(answer.lower().split())
    snips = set(' '.join(snippets).lower().split())
    if not toks: return 0.0
    return len(toks & snips) / len(toks)


def hallucination_penalty(answer: str, snippets: List[str]) -> float:
    ov = citation_overlap(answer, snippets)
    return 1.0 - ov


def aggregate_metrics(records: List[Dict]):
    # each record: {pred, gold, snippets, latency}
    overlaps = [citation_overlap(r['pred'], r.get('snippets', [])) for r in records]
    halluc = [hallucination_penalty(r['pred'], r.get('snippets', [])) for r in records]
    lat = [r.get('latency', 0.0) for r in records]
    acc = np.mean([float(r['pred'].strip().lower() == r['gold'].strip().lower()) for r in records])
    return {
        'accuracy': float(acc),
        'avg_citation_overlap': float(np.mean(overlaps)),
        'avg_hallucination_penalty': float(np.mean(halluc)),
        'avg_latency_sec': float(np.mean(lat)),
        'n': len(records)
    }
