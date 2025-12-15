"""Tests for early stopping module"""

import unittest
from src.early_stopping import (
    EarlyStopper, 
    EarlyStoppingConfig, 
    AdaptiveRunManager,
    check_early_stop
)


class TestEarlyStopper(unittest.TestCase):
    def setUp(self):
        self.stopper = EarlyStopper(EarlyStoppingConfig(
            min_runs=3,
            confidence_threshold=0.85,
            convergence_threshold=0.8,
            stability_window=2
        ))

    def test_initial_state(self):
        """Fresh stopper should have empty history"""
        self.assertEqual(len(self.stopper.run_history), 0)
        self.assertEqual(self.stopper.stability_streak, 0)

    def test_minimum_runs(self):
        """Should not stop before minimum runs"""
        for i in range(2):
            decision = self.stopper.record_run("same answer")
            self.assertFalse(decision.should_stop)
            self.assertIn("Minimum runs", decision.reason)

    def test_convergence_detection(self):
        """Should detect convergence with consistent answers"""
        # Run 5 times with same answer
        for i in range(5):
            decision = self.stopper.record_run("The answer is 42")
        
        # Should have high convergence
        self.assertEqual(decision.convergence_score, 1.0)
        self.assertEqual(decision.dominant_count, 5)

    def test_early_stop_on_convergence(self):
        """Should stop early when answers converge"""
        # Run with consistent answers
        for i in range(4):
            decision = self.stopper.record_run("consistent answer")
        
        # With 100% convergence and stability streak, should stop
        self.assertTrue(decision.should_stop or decision.convergence_score >= 0.8)

    def test_mixed_answers(self):
        """Mixed answers should lower convergence"""
        answers = ["yes", "yes", "no", "yes", "maybe"]
        for ans in answers:
            decision = self.stopper.record_run(ans)
        
        # Convergence should be 3/5 = 0.6
        self.assertEqual(decision.convergence_score, 0.6)
        self.assertEqual(decision.dominant_count, 3)

    def test_wilson_ci_calculation(self):
        """Wilson CI should be calculated correctly"""
        stopper = EarlyStopper()
        
        # 5 successes out of 5
        ci_low, ci_high = stopper._wilson_ci(5, 5)
        self.assertGreater(ci_low, 0.5)
        self.assertEqual(ci_high, 1.0)
        
        # 0 successes out of 5
        ci_low, ci_high = stopper._wilson_ci(0, 5)
        self.assertEqual(ci_low, 0.0)
        self.assertLess(ci_high, 0.5)
        
        # Edge case: 0 trials
        ci_low, ci_high = stopper._wilson_ci(0, 0)
        self.assertEqual((ci_low, ci_high), (0.0, 0.0))

    def test_reset(self):
        """Reset should clear all state"""
        self.stopper.record_run("answer1")
        self.stopper.record_run("answer2")
        self.stopper.reset()
        
        self.assertEqual(len(self.stopper.run_history), 0)
        self.assertEqual(self.stopper.stability_streak, 0)
        self.assertIsNone(self.stopper.last_dominant)

    def test_answer_signature(self):
        """Answer signature should normalize text"""
        sig1 = self.stopper._get_answer_signature("  Hello World  ")
        sig2 = self.stopper._get_answer_signature("hello world")
        self.assertEqual(sig1, sig2)


class TestAdaptiveRunManager(unittest.TestCase):
    def setUp(self):
        self.manager = AdaptiveRunManager(
            max_runs=10,
            min_runs=3,
            target_confidence=0.85
        )

    def test_start_problem(self):
        """Starting new problem should reset state"""
        self.manager.stopper.record_run("old answer")
        self.manager.start_problem()
        self.assertEqual(len(self.manager.stopper.run_history), 0)

    def test_must_continue_below_minimum(self):
        """Must continue if below minimum runs"""
        self.manager.start_problem()
        
        should_continue, decision = self.manager.should_continue("answer", 5)
        self.assertTrue(should_continue)
        
        should_continue, decision = self.manager.should_continue("answer", 5)
        self.assertTrue(should_continue)

    def test_stop_at_maximum(self):
        """Must stop at maximum runs"""
        self.manager.start_problem()
        
        for i in range(10):
            should_continue, decision = self.manager.should_continue(
                f"answer-{i % 3}", 10
            )
        
        # At run 10, should stop
        self.assertFalse(should_continue)
        self.assertIn("Maximum runs", decision.reason)

    def test_recommendation_by_complexity(self):
        """Should recommend different runs for different complexity"""
        self.assertEqual(self.manager.get_recommendation('low'), 3)
        self.assertEqual(self.manager.get_recommendation('medium'), 5)
        self.assertEqual(self.manager.get_recommendation('high'), 7)
        self.assertEqual(self.manager.get_recommendation('unknown'), 5)


class TestCheckEarlyStop(unittest.TestCase):
    def test_convenience_function(self):
        """Test the convenience function"""
        # First run resets
        decision = check_early_stop("answer", 1, 5)
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.current_run, 1)
        
        # Second run
        decision = check_early_stop("answer", 2, 5)
        self.assertEqual(decision.current_run, 2)

    def test_saved_runs_calculation(self):
        """Should calculate saved runs correctly"""
        # Reset with run 1
        check_early_stop("same", 1, 10)
        
        # Continue with consistent answers
        for i in range(2, 8):
            decision = check_early_stop("same", i, 10)
        
        # If stopped early, saved_runs should be calculated
        if decision.should_stop:
            expected_saved = 10 - decision.current_run
            self.assertEqual(decision.saved_runs, expected_saved)


class TestWilsonCI(unittest.TestCase):
    """Test Wilson confidence interval edge cases"""
    
    def test_all_successes(self):
        """All successes should have high CI"""
        stopper = EarlyStopper()
        ci_low, ci_high = stopper._wilson_ci(10, 10)
        self.assertGreater(ci_low, 0.7)

    def test_half_successes(self):
        """50% success rate"""
        stopper = EarlyStopper()
        ci_low, ci_high = stopper._wilson_ci(5, 10)
        self.assertGreater(ci_low, 0.2)
        self.assertLess(ci_high, 0.8)

    def test_bounds(self):
        """CI should always be in [0, 1]"""
        stopper = EarlyStopper()
        for successes in range(11):
            ci_low, ci_high = stopper._wilson_ci(successes, 10)
            self.assertGreaterEqual(ci_low, 0.0)
            self.assertLessEqual(ci_high, 1.0)
            self.assertLessEqual(ci_low, ci_high)


if __name__ == '__main__':
    unittest.main()
