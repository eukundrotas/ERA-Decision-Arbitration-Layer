"""
Adaptive Early Stopping Module
Level 1 Upgrade: Optimize number of runs based on convergence

Monitors answer stability across runs and stops early when
sufficient confidence is achieved, saving API costs and time.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class StoppingDecision:
    """Result of early stopping check"""
    should_stop: bool
    reason: str
    current_run: int
    total_planned_runs: int
    convergence_score: float  # 0.0-1.0, higher = more stable
    confidence_interval: Tuple[float, float]
    dominant_answer: Optional[str]
    dominant_count: int
    saved_runs: int


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping behavior"""
    min_runs: int = 3  # Minimum runs before considering early stop
    confidence_threshold: float = 0.85  # CI lower bound to trigger stop
    convergence_threshold: float = 0.8  # Proportion of same answer needed
    stability_window: int = 2  # Consecutive stable runs needed
    enable_adaptive: bool = True  # Allow dynamic threshold adjustment


class EarlyStopper:
    """
    Monitors run stability and decides when to stop early.
    
    Uses Wilson score interval and answer convergence to determine
    when additional runs are unlikely to change the outcome.
    """
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.run_history: List[str] = []
        self.stability_streak: int = 0
        self.last_dominant: Optional[str] = None
    
    def reset(self):
        """Reset state for a new problem"""
        self.run_history = []
        self.stability_streak = 0
        self.last_dominant = None
    
    def _wilson_ci(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        
        Args:
            successes: Number of successes (dominant answer count)
            total: Total trials (runs)
            confidence: Confidence level (default 95%)
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        if total == 0:
            return (0.0, 0.0)
        
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
        
        p = successes / total
        
        denominator = 1 + z * z / total
        center = p + z * z / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        
        lower = (center - spread) / denominator
        upper = (center + spread) / denominator
        
        return (max(0.0, lower), min(1.0, upper))
    
    def _get_answer_signature(self, answer: str) -> str:
        """
        Create a normalized signature for answer comparison.
        Uses first 200 chars, lowercased, stripped.
        """
        if not answer:
            return ""
        return answer[:200].lower().strip()
    
    def record_run(self, answer: str) -> StoppingDecision:
        """
        Record a run result and check if we should stop.
        
        Args:
            answer: The final answer from this run
            
        Returns:
            StoppingDecision with recommendation
        """
        signature = self._get_answer_signature(answer)
        self.run_history.append(signature)
        
        current_run = len(self.run_history)
        
        # Count answer frequencies
        counter = Counter(self.run_history)
        dominant_answer, dominant_count = counter.most_common(1)[0]
        
        # Calculate convergence score
        convergence = dominant_count / current_run
        
        # Calculate Wilson CI
        ci_lower, ci_upper = self._wilson_ci(dominant_count, current_run)
        
        # Check stability streak
        if dominant_answer == self.last_dominant:
            self.stability_streak += 1
        else:
            self.stability_streak = 1
            self.last_dominant = dominant_answer
        
        # Default: don't stop
        should_stop = False
        reason = "Continuing runs"
        
        # Check stopping conditions
        if current_run < self.config.min_runs:
            reason = f"Minimum runs not reached ({current_run}/{self.config.min_runs})"
        
        elif ci_lower >= self.config.confidence_threshold:
            should_stop = True
            reason = f"High confidence achieved (CI lower={ci_lower:.3f} >= {self.config.confidence_threshold})"
        
        elif convergence >= self.config.convergence_threshold and \
             self.stability_streak >= self.config.stability_window:
            should_stop = True
            reason = f"Stable convergence ({convergence:.1%} agreement, {self.stability_streak} streak)"
        
        elif self.config.enable_adaptive and current_run >= 5:
            # Adaptive: if we've done 5+ runs and have 80%+ agreement
            if convergence >= 0.8:
                should_stop = True
                reason = f"Adaptive stop: {convergence:.1%} agreement after {current_run} runs"
        
        return StoppingDecision(
            should_stop=should_stop,
            reason=reason,
            current_run=current_run,
            total_planned_runs=0,  # Will be set by caller
            convergence_score=convergence,
            confidence_interval=(ci_lower, ci_upper),
            dominant_answer=dominant_answer,
            dominant_count=dominant_count,
            saved_runs=0  # Will be calculated by caller
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        if not self.run_history:
            return {"runs": 0, "convergence": 0.0, "dominant": None}
        
        counter = Counter(self.run_history)
        dominant, count = counter.most_common(1)[0]
        
        return {
            "runs": len(self.run_history),
            "convergence": count / len(self.run_history),
            "dominant_count": count,
            "unique_answers": len(counter),
            "stability_streak": self.stability_streak,
            "ci": self._wilson_ci(count, len(self.run_history))
        }


class AdaptiveRunManager:
    """
    High-level manager for adaptive run scheduling.
    
    Decides how many runs to execute based on problem complexity
    and observed answer stability.
    """
    
    def __init__(
        self,
        max_runs: int = 10,
        min_runs: int = 3,
        target_confidence: float = 0.85
    ):
        self.max_runs = max_runs
        self.min_runs = min_runs
        self.target_confidence = target_confidence
        self.stopper = EarlyStopper(EarlyStoppingConfig(
            min_runs=min_runs,
            confidence_threshold=target_confidence,
            enable_adaptive=True
        ))
    
    def start_problem(self):
        """Initialize for a new problem"""
        self.stopper.reset()
    
    def should_continue(self, answer: str, planned_total: int) -> Tuple[bool, StoppingDecision]:
        """
        Check if we should continue running.
        
        Args:
            answer: Latest run's answer
            planned_total: Originally planned total runs
            
        Returns:
            (should_continue, decision) tuple
        """
        decision = self.stopper.record_run(answer)
        decision.total_planned_runs = planned_total
        
        current = decision.current_run
        
        # Must continue if below minimum
        if current < self.min_runs:
            return True, decision
        
        # Must stop if at maximum
        if current >= self.max_runs:
            decision.should_stop = True
            decision.reason = f"Maximum runs reached ({self.max_runs})"
            return False, decision
        
        # Early stop check
        if decision.should_stop:
            decision.saved_runs = planned_total - current
            logger.info(f"Early stopping: {decision.reason} (saved {decision.saved_runs} runs)")
            return False, decision
        
        return True, decision
    
    def get_recommendation(self, problem_complexity: str = "medium") -> int:
        """
        Get recommended number of runs based on complexity.
        
        Args:
            problem_complexity: 'low', 'medium', or 'high'
            
        Returns:
            Recommended number of runs
        """
        base_runs = {
            'low': 3,
            'medium': 5,
            'high': 7
        }
        return base_runs.get(problem_complexity, 5)


# Singleton instances
early_stopper = EarlyStopper()
run_manager = AdaptiveRunManager()


def check_early_stop(answer: str, current_run: int, total_runs: int) -> StoppingDecision:
    """
    Convenience function to check if we should stop early.
    
    Args:
        answer: Latest answer
        current_run: Current run number (1-indexed)
        total_runs: Total planned runs
        
    Returns:
        StoppingDecision
    """
    # Reset if this is run 1
    if current_run == 1:
        early_stopper.reset()
    
    decision = early_stopper.record_run(answer)
    decision.total_planned_runs = total_runs
    decision.saved_runs = total_runs - current_run if decision.should_stop else 0
    
    return decision
