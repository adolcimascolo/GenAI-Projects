# Autoregressive LLM Evaluation Harness

Benchmarks open-source LLMs (Llama/Mistral class) on biomedical QA with retrieval (FAISS), reranking, hallucination scoring, and latency/cost logging.

## Datasets
- PubMedQA (public biomedical QA): https://pubmedqa.github.io/
- Optional: PubMed abstracts via PMC-OA (for retrieval corpora): https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/

## Quickstart
```bash
# 1) Create env (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Prepare a small subset (place your PubMedQA JSON in data/pubmedqa.json)
python src/data.py --prepare --input data/pubmedqa.json --output data/eval_samples.jsonl

# 3) Build FAISS index from contexts
python src/data.py --build-index --corpus data/corpus.txt --index data/faiss.index --embeddings data/embeddings.npy

# 4) Run evaluation on a small model
python src/evaluate.py --model meta-llama/Llama-3-8b-instruct --samples data/eval_samples.jsonl --index data/faiss.index --embeddings data/embeddings.npy --report reports/llama3_eval.json
```

> Tip: Replace the model with any local HF path or GGUF quantization and wire your own inference in `src/evaluate.py`.

## What this project demonstrates
- Evidence-grounded QA (retrieval + reranking)
- Hallucination scoring against cited snippets
- Throughput/latency and token-cost tracking

## Repo structure
```
src/
  config.py
  data.py
  metrics.py
  evaluate.py
  utils.py
data/
  (place datasets here)
reports/
```
