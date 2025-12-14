"""Тесты для Orchestrator"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import MultiLLMOrchestrator
from src.stability import StabilityAnalyzer, WilsonCI

class TestOrchestrator(unittest.TestCase):
    
    def test_initialization(self):
        orchestrator = MultiLLMOrchestrator(
            pool_name="science",
            domain="science",
            out_dir="./test_out"
        )
        self.assertEqual(len(orchestrator.pool.solvers), 7)
        self.assertEqual(orchestrator.domain, "science")
    
    def test_model_memory_load(self):
        orchestrator = MultiLLMOrchestrator(
            pool_name="science",
            domain="science",
            out_dir="./test_out"
        )
        memory = orchestrator.model_memory
        self.assertIsNotNone(memory.state)

class TestStabilityAnalyzer(unittest.TestCase):
    
    def test_stability_analysis_empty(self):
        analyzer = StabilityAnalyzer()
        result = analyzer.analyze([])
        self.assertEqual(result["majority_rate"], 0.0)
        self.assertIsNone(result["majority_mode"])
    
    def test_stability_analysis_unanimous(self):
        analyzer = StabilityAnalyzer()
        results = [
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
        ]
        result = analyzer.analyze(results)
        self.assertEqual(result["majority_rate"], 1.0)
        self.assertEqual(result["majority_mode"], "hard_select")
    
    def test_stability_analysis_mixed(self):
        analyzer = StabilityAnalyzer()
        results = [
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
            {"decision_mode": "hard_select", "used_candidates": ["model_02"]},
            {"decision_mode": "consensus_top3", "used_candidates": ["model_01"]},
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
            {"decision_mode": "hard_select", "used_candidates": ["model_01"]},
        ]
        result = analyzer.analyze(results)
        self.assertEqual(result["majority_rate"], 0.8)
        self.assertEqual(result["majority_mode"], "hard_select")

class TestWilsonCI(unittest.TestCase):
    
    def test_wilson_ci_zero_trials(self):
        lower, upper = WilsonCI.calculate(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)
    
    def test_wilson_ci_all_successes(self):
        lower, upper = WilsonCI.calculate(5, 5)
        self.assertGreater(lower, 0.5)
        self.assertEqual(upper, 1.0)
    
    def test_wilson_ci_half_successes(self):
        lower, upper = WilsonCI.calculate(3, 6)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)

if __name__ == '__main__':
    unittest.main()
