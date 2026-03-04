from pydantic import BaseModel
from typing import List, Dict

class Plan(BaseModel):
    subqueries: List[str]

class Evidence(BaseModel):
    snippets: List[str]

class Answer(BaseModel):
    text: str
    citations: List[str]
    low_confidence: bool = False

class QueryPlanner:
    def plan(self, question: str) -> Plan:
        # naive split heuristic for demo
        parts = [p.strip() for p in question.split(' vs ') if p.strip()]
        subs = [f"mechanism of action for {p}" for p in parts]
        subs += [f"safety profile for {p}" for p in parts]
        return Plan(subqueries=subs or [question])

class ComplianceGuard:
    def check(self, answer: Answer) -> Answer:
        low = len(answer.citations) < 2 or any(len(c) < 40 for c in answer.citations)
        answer.low_confidence = low
        return answer
