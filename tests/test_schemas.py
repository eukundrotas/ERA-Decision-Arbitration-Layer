"""Тесты для JSON schemas"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import validate_solver_json, validate_arbiter_json

class TestSolverSchema(unittest.TestCase):
    
    def test_valid_solver_json(self):
        valid_solver = {
            "model_id": "test-model",
            "task_id": "task_001",
            "final_answer": "Test answer",
            "confidence": 0.85,
            "assumptions": ["A1"],
            "risks": ["R1"],
            "evidence": ["E1"],
            "self_checks": ["S1"]
        }
        # Should not raise
        validate_solver_json(valid_solver)
    
    def test_invalid_confidence(self):
        invalid_solver = {
            "model_id": "test-model",
            "task_id": "task_001",
            "final_answer": "Test answer",
            "confidence": 1.5,  # Invalid: > 1
            "assumptions": [],
            "risks": [],
            "evidence": [],
            "self_checks": []
        }
        with self.assertRaises(ValueError):
            validate_solver_json(invalid_solver)
    
    def test_missing_field(self):
        invalid_solver = {
            "model_id": "test-model",
            "task_id": "task_001",
            "final_answer": "Test answer"
            # Missing: confidence, assumptions, risks, evidence, self_checks
        }
        with self.assertRaises(ValueError):
            validate_solver_json(invalid_solver)

class TestArbiterSchema(unittest.TestCase):
    
    def test_valid_arbiter_json(self):
        valid_arbiter = {
            "task_id": "task_001",
            "selected_model_id": "model_01",
            "ranking": [
                {"model_id": "model_01", "score": 0.91},
                {"model_id": "model_02", "score": 0.85}
            ],
            "final_answer": "Best answer",
            "arbiter_notes": ["Note 1"]
        }
        # Should not raise
        validate_arbiter_json(valid_arbiter)
    
    def test_invalid_ranking_score(self):
        invalid_arbiter = {
            "task_id": "task_001",
            "selected_model_id": "model_01",
            "ranking": [
                {"model_id": "model_01", "score": 1.5}  # Invalid: > 1
            ],
            "final_answer": "Answer",
            "arbiter_notes": []
        }
        with self.assertRaises(ValueError):
            validate_arbiter_json(invalid_arbiter)

if __name__ == '__main__':
    unittest.main()
