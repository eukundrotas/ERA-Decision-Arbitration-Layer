"""Tests for dashboard API module"""

import unittest
from src.api import DashboardTracker, RunEvent, ProblemSession


class TestDashboardTracker(unittest.TestCase):
    def setUp(self):
        # Get fresh tracker instance
        self.tracker = DashboardTracker()
        # Reset state for clean tests
        self.tracker.sessions = {}
        self.tracker.recent_events.clear()
        self.tracker.model_stats = {}
        self.tracker.global_stats = {
            "total_problems": 0,
            "total_runs": 0,
            "total_api_calls": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "started_at": "2024-01-01T00:00:00"
        }

    def test_start_session(self):
        """Should create a new session"""
        problem_id = self.tracker.start_session(
            problem_text="What is 2+2?",
            pool_name="math",
            total_runs=5
        )
        
        self.assertIsNotNone(problem_id)
        self.assertTrue(problem_id.startswith("prob_"))
        self.assertIn(problem_id, self.tracker.sessions)
        self.assertEqual(self.tracker.global_stats["total_problems"], 1)

    def test_record_solver_complete(self):
        """Should record solver completions"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=3
        )
        
        self.tracker.record_solver_complete(
            problem_id=problem_id,
            run_number=1,
            model_id="gpt-4",
            answer_preview="The answer is...",
            confidence=0.9,
            latency_ms=500
        )
        
        self.assertEqual(self.tracker.global_stats["total_api_calls"], 1)
        self.assertIn("gpt-4", self.tracker.model_stats)
        self.assertEqual(self.tracker.model_stats["gpt-4"]["calls"], 1)

    def test_record_run_complete(self):
        """Should record run completions"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=3
        )
        
        self.tracker.record_run_complete(
            problem_id=problem_id,
            run_number=1,
            decision_mode="hard_select",
            final_answer_preview="Final answer"
        )
        
        self.assertEqual(self.tracker.sessions[problem_id].completed_runs, 1)
        self.assertEqual(self.tracker.global_stats["total_runs"], 1)

    def test_end_session(self):
        """Should properly end a session"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=3
        )
        
        self.tracker.end_session(
            problem_id=problem_id,
            status="completed",
            final_result={"answer": "42"}
        )
        
        session = self.tracker.sessions[problem_id]
        self.assertEqual(session.status, "completed")
        self.assertIsNotNone(session.ended_at)
        self.assertEqual(session.final_result["answer"], "42")

    def test_get_session(self):
        """Should return session details"""
        problem_id = self.tracker.start_session(
            problem_text="What is gravity?",
            pool_name="science",
            total_runs=5
        )
        
        session = self.tracker.get_session(problem_id)
        
        self.assertIsNotNone(session)
        self.assertEqual(session["problem_id"], problem_id)
        self.assertEqual(session["pool_name"], "science")
        self.assertEqual(session["progress"], "0/5")

    def test_get_nonexistent_session(self):
        """Should return None for nonexistent session"""
        session = self.tracker.get_session("nonexistent")
        self.assertIsNone(session)

    def test_get_recent_events(self):
        """Should return recent events"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=3
        )
        
        events = self.tracker.get_recent_events(limit=10)
        
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1]["event_type"], "start")

    def test_get_model_stats(self):
        """Should return model statistics"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=3
        )
        
        # Record some model calls
        for i in range(3):
            self.tracker.record_solver_complete(
                problem_id=problem_id,
                run_number=1,
                model_id="claude-3",
                answer_preview="Answer",
                confidence=0.8 + i * 0.05,
                latency_ms=400 + i * 100
            )
        
        stats = self.tracker.get_model_stats()
        
        self.assertIn("claude-3", stats)
        self.assertEqual(stats["claude-3"]["calls"], 3)
        self.assertGreater(stats["claude-3"]["avg_latency_ms"], 0)

    def test_get_dashboard_summary(self):
        """Should return complete dashboard summary"""
        # Create some activity
        problem_id = self.tracker.start_session(
            problem_text="Test problem",
            pool_name="math",
            total_runs=5
        )
        
        self.tracker.record_solver_complete(
            problem_id=problem_id,
            run_number=1,
            model_id="gpt-4",
            answer_preview="Answer",
            confidence=0.9,
            latency_ms=500
        )
        
        summary = self.tracker.get_dashboard_summary()
        
        self.assertEqual(summary["status"], "healthy")
        self.assertEqual(summary["total_problems"], 1)
        self.assertIn("model_stats", summary)
        self.assertIn("uptime_since", summary)

    def test_event_limit(self):
        """Recent events should be limited"""
        problem_id = self.tracker.start_session(
            problem_text="Test",
            pool_name="science",
            total_runs=100
        )
        
        # Generate many events
        for i in range(50):
            self.tracker.record_solver_complete(
                problem_id=problem_id,
                run_number=i,
                model_id=f"model-{i % 5}",
                answer_preview="Answer",
                confidence=0.8,
                latency_ms=500
            )
        
        # Should only return requested limit
        events = self.tracker.get_recent_events(limit=10)
        self.assertEqual(len(events), 10)


class TestRunEvent(unittest.TestCase):
    def test_event_creation(self):
        """Should create RunEvent properly"""
        event = RunEvent(
            timestamp="2024-01-01T12:00:00",
            event_type="solver_complete",
            problem_id="prob_123",
            run_number=1,
            data={"model_id": "gpt-4"}
        )
        
        self.assertEqual(event.event_type, "solver_complete")
        self.assertEqual(event.run_number, 1)
        self.assertEqual(event.data["model_id"], "gpt-4")


class TestProblemSession(unittest.TestCase):
    def test_session_creation(self):
        """Should create ProblemSession properly"""
        session = ProblemSession(
            problem_id="prob_test",
            problem_text="Test problem",
            pool_name="science",
            started_at="2024-01-01T12:00:00",
            total_runs=5
        )
        
        self.assertEqual(session.status, "running")
        self.assertEqual(session.completed_runs, 0)
        self.assertIsNone(session.ended_at)


if __name__ == '__main__':
    unittest.main()
