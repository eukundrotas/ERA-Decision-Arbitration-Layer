"""
A/B Testing Framework
Level 2 Upgrade: Compare model configurations and strategies

Provides infrastructure for:
- Configuration experiments
- Model comparison tests
- Strategy evaluation
- Statistical significance testing
"""

import json
import math
import random
import logging
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment"""
    name: str
    description: str
    variants: Dict[str, Dict[str, Any]]  # variant_name -> config overrides
    traffic_split: Dict[str, float]  # variant_name -> percentage (0.0-1.0)
    metrics: List[str]  # metrics to track
    min_samples: int = 30  # minimum samples per variant
    confidence_level: float = 0.95  # for significance testing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "draft"  # 'draft', 'running', 'paused', 'completed'


@dataclass
class ExperimentResult:
    """Result of a single experiment run"""
    experiment_id: str
    variant: str
    problem_id: str
    timestamp: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStatistics:
    """Statistics for a single variant"""
    variant: str
    sample_count: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class ExperimentAnalysis:
    """Complete analysis of an experiment"""
    experiment_id: str
    experiment_name: str
    total_samples: int
    variant_stats: Dict[str, VariantStatistics]
    winner: Optional[str]
    winner_confidence: float
    is_significant: bool
    recommendation: str
    details: Dict[str, Any]


class ABExperiment:
    """
    Single A/B experiment manager.
    """
    
    def __init__(self, config: ExperimentConfig, experiment_id: Optional[str] = None):
        self.config = config
        self.experiment_id = experiment_id or self._generate_id()
        self.results: List[ExperimentResult] = []
        self._variant_counts: Dict[str, int] = defaultdict(int)
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID"""
        hash_input = f"{self.config.name}{datetime.now().isoformat()}"
        return f"exp_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def assign_variant(self, user_id: Optional[str] = None) -> str:
        """
        Assign a variant based on traffic split.
        
        Uses consistent hashing if user_id provided (same user gets same variant).
        """
        if user_id:
            # Consistent assignment based on user_id
            hash_val = int(hashlib.md5(f"{self.experiment_id}{user_id}".encode()).hexdigest(), 16)
            threshold = hash_val / (2 ** 128)
        else:
            threshold = random.random()
        
        cumulative = 0.0
        for variant, split in self.config.traffic_split.items():
            cumulative += split
            if threshold < cumulative:
                return variant
        
        # Fallback to first variant
        return list(self.config.variants.keys())[0]
    
    def get_variant_config(self, variant: str) -> Dict[str, Any]:
        """Get configuration overrides for a variant"""
        return self.config.variants.get(variant, {})
    
    def record_result(
        self,
        variant: str,
        problem_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """Record a result for this experiment"""
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            variant=variant,
            problem_id=problem_id,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        self._variant_counts[variant] += 1
        
        return result
    
    def get_variant_statistics(self, variant: str) -> VariantStatistics:
        """Calculate statistics for a variant"""
        variant_results = [r for r in self.results if r.variant == variant]
        
        if not variant_results:
            return VariantStatistics(
                variant=variant,
                sample_count=0,
                metrics={},
                confidence_intervals={}
            )
        
        # Aggregate metrics
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for result in variant_results:
            for metric, value in result.metrics.items():
                metric_values[metric].append(value)
        
        # Calculate stats
        metrics_stats = {}
        confidence_intervals = {}
        
        for metric, values in metric_values.items():
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
            std = math.sqrt(variance)
            
            metrics_stats[metric] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "count": n
            }
            
            # 95% confidence interval
            z = 1.96  # 95% confidence
            margin = z * std / math.sqrt(n) if n > 0 else 0
            confidence_intervals[metric] = (mean - margin, mean + margin)
        
        return VariantStatistics(
            variant=variant,
            sample_count=len(variant_results),
            metrics=metrics_stats,
            confidence_intervals=confidence_intervals
        )
    
    def analyze(self) -> ExperimentAnalysis:
        """
        Analyze experiment results and determine winner.
        """
        variants = list(self.config.variants.keys())
        variant_stats = {v: self.get_variant_statistics(v) for v in variants}
        
        total_samples = sum(s.sample_count for s in variant_stats.values())
        
        # Check if we have enough samples
        has_enough_samples = all(
            s.sample_count >= self.config.min_samples 
            for s in variant_stats.values()
        )
        
        # Find winner for primary metric (first in list)
        primary_metric = self.config.metrics[0] if self.config.metrics else "score"
        
        winner = None
        winner_confidence = 0.0
        is_significant = False
        
        if len(variants) >= 2 and has_enough_samples:
            # Compare variants pairwise
            best_variant = None
            best_mean = -float('inf')
            
            for variant, stats in variant_stats.items():
                if primary_metric in stats.metrics:
                    mean = stats.metrics[primary_metric]["mean"]
                    if mean > best_mean:
                        best_mean = mean
                        best_variant = variant
            
            if best_variant:
                # Calculate statistical significance
                winner = best_variant
                
                # Simple significance test: check if confidence intervals don't overlap
                winner_ci = variant_stats[winner].confidence_intervals.get(primary_metric, (0, 0))
                
                others_max_ci_upper = max(
                    variant_stats[v].confidence_intervals.get(primary_metric, (0, 0))[1]
                    for v in variants if v != winner
                )
                
                is_significant = winner_ci[0] > others_max_ci_upper
                winner_confidence = 0.95 if is_significant else 0.7
        
        # Generate recommendation
        if not has_enough_samples:
            recommendation = f"Continue experiment: need {self.config.min_samples} samples per variant"
        elif not is_significant:
            recommendation = f"No clear winner yet. Consider running longer or increasing sample size."
        else:
            recommendation = f"Recommend {winner} as winner with {winner_confidence:.0%} confidence"
        
        return ExperimentAnalysis(
            experiment_id=self.experiment_id,
            experiment_name=self.config.name,
            total_samples=total_samples,
            variant_stats={v: asdict(s) for v, s in variant_stats.items()},
            winner=winner,
            winner_confidence=winner_confidence,
            is_significant=is_significant,
            recommendation=recommendation,
            details={
                "has_enough_samples": has_enough_samples,
                "primary_metric": primary_metric,
                "min_samples_required": self.config.min_samples,
                "variants": variants
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize experiment to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "variant_counts": dict(self._variant_counts)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABExperiment':
        """Deserialize experiment from dictionary"""
        config = ExperimentConfig(**data["config"])
        experiment = cls(config, data["experiment_id"])
        experiment.results = [ExperimentResult(**r) for r in data["results"]]
        experiment._variant_counts = defaultdict(int, data.get("variant_counts", {}))
        return experiment


class ABTestingManager:
    """
    Central manager for all A/B experiments.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.experiments: Dict[str, ABExperiment] = {}
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments from storage"""
        if not self.storage_path:
            return
        
        for file in self.storage_path.glob("exp_*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    experiment = ABExperiment.from_dict(data)
                    self.experiments[experiment.experiment_id] = experiment
            except Exception as e:
                logger.error(f"Failed to load experiment {file}: {e}")
    
    def _save_experiment(self, experiment: ABExperiment):
        """Save experiment to storage"""
        if not self.storage_path:
            return
        
        file_path = self.storage_path / f"{experiment.experiment_id}.json"
        with open(file_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        metrics: Optional[List[str]] = None,
        min_samples: int = 30
    ) -> ABExperiment:
        """
        Create a new A/B experiment.
        
        Args:
            name: Experiment name
            description: What this experiment tests
            variants: Dict of variant_name -> config overrides
            traffic_split: Traffic allocation (default: equal split)
            metrics: Metrics to track (default: ['score'])
            min_samples: Minimum samples per variant
        """
        # Default to equal traffic split
        if traffic_split is None:
            split_value = 1.0 / len(variants)
            traffic_split = {v: split_value for v in variants}
        
        config = ExperimentConfig(
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            metrics=metrics or ["score"],
            min_samples=min_samples,
            start_time=datetime.now().isoformat(),
            status="running"
        )
        
        experiment = ABExperiment(config)
        self.experiments[experiment.experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment {experiment.experiment_id}: {name}")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status"""
        result = []
        for exp in self.experiments.values():
            if status and exp.config.status != status:
                continue
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.config.name,
                "status": exp.config.status,
                "total_results": len(exp.results),
                "variants": list(exp.config.variants.keys())
            })
        return result
    
    def record_result(
        self,
        experiment_id: str,
        variant: str,
        problem_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentResult]:
        """Record a result for an experiment"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return None
        
        result = experiment.record_result(variant, problem_id, metrics, metadata)
        self._save_experiment(experiment)
        return result
    
    def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentAnalysis]:
        """Analyze an experiment"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
        return experiment.analyze()
    
    def complete_experiment(self, experiment_id: str) -> Optional[ExperimentAnalysis]:
        """Mark experiment as completed and return final analysis"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        experiment.config.status = "completed"
        experiment.config.end_time = datetime.now().isoformat()
        self._save_experiment(experiment)
        
        return experiment.analyze()


# Pre-configured experiments for ERA DAL
def create_model_pool_experiment(manager: ABTestingManager) -> ABExperiment:
    """Create experiment comparing different model pool configurations"""
    return manager.create_experiment(
        name="Model Pool Size Comparison",
        description="Compare 5 vs 7 vs 9 models in solver pool",
        variants={
            "small_pool": {"solver_pool_size": 5},
            "medium_pool": {"solver_pool_size": 7},
            "large_pool": {"solver_pool_size": 9}
        },
        metrics=["stability", "latency", "cost", "quality"],
        min_samples=50
    )


def create_consensus_strategy_experiment(manager: ABTestingManager) -> ABExperiment:
    """Create experiment comparing consensus strategies"""
    return manager.create_experiment(
        name="Consensus Strategy Comparison",
        description="Compare hard_select vs consensus_top2 vs consensus_top3",
        variants={
            "hard_select": {"consensus_mode": "hard_select", "consensus_topk": 1},
            "consensus_top2": {"consensus_mode": "consensus", "consensus_topk": 2},
            "consensus_top3": {"consensus_mode": "consensus", "consensus_topk": 3}
        },
        metrics=["stability", "quality", "agreement_rate"],
        min_samples=30
    )


def create_temperature_experiment(manager: ABTestingManager) -> ABExperiment:
    """Create experiment comparing temperature settings"""
    return manager.create_experiment(
        name="Temperature Comparison",
        description="Compare low vs medium vs high temperature for solvers",
        variants={
            "low_temp": {"temperature": 0.3},
            "medium_temp": {"temperature": 0.7},
            "high_temp": {"temperature": 1.0}
        },
        metrics=["diversity", "quality", "stability"],
        min_samples=40
    )


# Singleton manager instance
ab_manager = ABTestingManager()
