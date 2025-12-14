"""Тесты для SolverPool"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver_pool import SolverPool, SolverSpec

class TestSolverPool(unittest.TestCase):
    
    def test_pool_initialization(self):
        pool = SolverPool("science")
        self.assertEqual(pool.pool_name, "science")
        self.assertEqual(len(pool.solvers), 7)
    
    def test_math_pool(self):
        pool = SolverPool("math")
        self.assertEqual(pool.pool_name, "math")
        self.assertEqual(len(pool.solvers), 6)
    
    def test_med_pool(self):
        pool = SolverPool("med")
        self.assertEqual(pool.pool_name, "med")
        self.assertEqual(len(pool.solvers), 5)
    
    def test_econ_pool(self):
        pool = SolverPool("econ")
        self.assertEqual(pool.pool_name, "econ")
        self.assertEqual(len(pool.solvers), 6)
    
    def test_unknown_pool_defaults_to_science(self):
        pool = SolverPool("unknown")
        self.assertEqual(len(pool.solvers), 7)  # science pool size

class TestSolverSpec(unittest.TestCase):
    
    def test_solver_spec_creation(self):
        spec = SolverSpec("test-model", 0.5)
        self.assertEqual(spec.model_id, "test-model")
        self.assertEqual(spec.temperature, 0.5)
    
    def test_solver_spec_default_temperature(self):
        spec = SolverSpec("test-model")
        self.assertEqual(spec.temperature, 0.7)

if __name__ == '__main__':
    unittest.main()
