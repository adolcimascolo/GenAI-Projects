from dataclasses import dataclass

@dataclass
class EvalConfig:
    model: str
    samples_path: str
    index_path: str
    embeddings_path: str
    report_path: str = 'reports/report.json'
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    k_retrieval: int = 5
