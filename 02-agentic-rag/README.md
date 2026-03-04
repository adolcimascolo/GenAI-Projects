# Agentic RAG over Enterprise-Style Corpora (LangGraph)

Multi-agent RAG system for scientific/medical corpora (PMC-OA PDFs, FDA DailyMed labels). Demonstrates planning, retrieval, evidence extraction, and compliance guardrails.

## Data Sources (public)
- PubMed Central Open-Access (PMC-OA): https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- DailyMed Drug Labels: https://dailymed.nlm.nih.gov/

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Convert your PDFs to text into data/corpus/
python src/retriever.py --ingest data/raw_pdfs --out data/corpus.txt

# Run the agentic pipeline
python src/app.py --question "Compare safety profiles of GLP-1 agonists vs DPP-4 inhibitors" --corpus data/corpus.txt --index data/faiss.index
```

## Agents
- QueryPlanner → decomposes question
- Retriever → FAISS + reranker
- EvidenceExtractor → structures findings & citations
- ComplianceGuard → enforces citation coverage / low-hallucination

## Output
- JSON with reasoning trace, citations, and confidence flags.
