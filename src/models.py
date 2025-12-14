from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class SolverOutput:
    """Выход от solver'а"""
    model_id: str
    task_id: str
    final_answer: str
    confidence: float
    assumptions: List[str]
    risks: List[str]
    evidence: List[str]
    self_checks: List[str]
    latency_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

@dataclass
class ArbiterRanking:
    """Ранжирование арбитром"""
    model_id: str
    score: float

@dataclass
class ArbiterOutput:
    """Выход от arbiter'а"""
    task_id: str
    selected_model_id: str
    ranking: List[ArbiterRanking]
    final_answer: str
    arbiter_notes: List[str]
    decision_mode: str = "hard_select"
    used_candidates: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['ranking'] = [asdict(r) for r in self.ranking]
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

@dataclass
class RunRecord:
    """Одна строка в runs.csv"""
    pool: str
    iter_index: int
    round_number: int
    solver_model_id: str
    latency_ms: int
    confidence: float
    arbiter_score: Optional[float]
    decision_mode: str
    used_candidates: str
    final_answer: str
    notes: str
    timestamp: str

@dataclass
class StabilityResult:
    """Результаты stability analysis"""
    majority_rate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    majority_mode: str
    mode_distribution: Dict[str, int]
    total_runs: int
