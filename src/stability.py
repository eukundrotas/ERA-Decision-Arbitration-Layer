import logging
from typing import List, Dict, Any
from collections import Counter
import math

logger = logging.getLogger(__name__)

class WilsonCI:
    """Доверительный интервал Wilson для majority_rate"""
    
    @staticmethod
    def calculate(successes: int, trials: int, confidence: float = 0.95) -> tuple:
        """Вычисляет 95% доверительный интервал"""
        if trials == 0:
            return 0.0, 0.0
        
        z = 1.96  # для 95%
        
        p_hat = successes / trials
        
        denominator = 1 + z**2 / trials
        centre_adjusted = (p_hat + z**2 / (2 * trials)) / denominator
        
        adjusted_sd = math.sqrt(
            (p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials
        ) / denominator
        
        lower = max(0.0, centre_adjusted - z * adjusted_sd)
        upper = min(1.0, centre_adjusted + z * adjusted_sd)
        
        return lower, upper

class StabilityAnalyzer:
    """Анализирует устойчивость результатов"""
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        results_per_run: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Анализирует multi-run результаты.
        results_per_run = [
            {"decision_mode": "hard_select", "used_candidates": ["solver_02"]},
            {"decision_mode": "hard_select", "used_candidates": ["solver_02"]},
            ...
        ]
        """
        
        if not results_per_run:
            return {
                "majority_rate": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "majority_mode": None,
                "mode_distribution": {}
            }
        
        # Считаем, что "consensus" достигнут, если decision_mode одинаковый
        modes = [r["decision_mode"] for r in results_per_run]
        mode_counter = Counter(modes)
        majority_mode = mode_counter.most_common(1)[0][0]
        majority_count = mode_counter.most_common(1)[0][1]
        
        majority_rate = majority_count / len(results_per_run)
        
        ci_lower, ci_upper = WilsonCI.calculate(majority_count, len(results_per_run))
        
        return {
            "majority_rate": round(majority_rate, 3),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),
            "majority_mode": majority_mode,
            "mode_distribution": dict(mode_counter),
            "total_runs": len(results_per_run)
        }
