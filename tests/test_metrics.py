"""Tests for multi-metric evaluation module"""

import unittest
from src.metrics import (
    TextMetrics, 
    MultiMetricEvaluator, 
    evaluate_answer,
    compare_answers
)


class TestTextMetrics(unittest.TestCase):
    
    def test_tokenize(self):
        """Test tokenization"""
        tokens = TextMetrics.tokenize("Hello, World!")
        self.assertEqual(tokens, ["hello", "world"])
    
    def test_get_ngrams(self):
        """Test n-gram extraction"""
        tokens = ["a", "b", "c", "d"]
        bigrams = TextMetrics.get_ngrams(tokens, 2)
        self.assertEqual(len(bigrams), 3)
        self.assertIn(("a", "b"), bigrams)
        self.assertIn(("c", "d"), bigrams)
    
    def test_bleu_identical(self):
        """BLEU score for identical texts should be high"""
        text = "The quick brown fox jumps over the lazy dog"
        result = TextMetrics.bleu_score(text, text)
        self.assertGreater(result.score, 0.9)
    
    def test_bleu_different(self):
        """BLEU score for different texts should be lower"""
        candidate = "The cat sat on the mat"
        reference = "Machine learning is fascinating"
        result = TextMetrics.bleu_score(candidate, reference)
        self.assertLess(result.score, 0.3)
    
    def test_rouge_identical(self):
        """ROUGE score for identical texts"""
        text = "The quick brown fox jumps"
        result = TextMetrics.rouge_score(text, text, "rouge-1")
        self.assertEqual(result.score, 1.0)
    
    def test_rouge_l(self):
        """ROUGE-L uses longest common subsequence"""
        candidate = "A B C D E"
        reference = "A X B Y C Z D E"
        result = TextMetrics.rouge_score(candidate, reference, "rouge-l")
        self.assertGreater(result.score, 0.5)
    
    def test_levenshtein_identical(self):
        """Identical texts should have similarity 1.0"""
        result = TextMetrics.levenshtein_similarity("hello world", "hello world")
        self.assertEqual(result.score, 1.0)
    
    def test_levenshtein_different(self):
        """Different texts should have lower similarity"""
        result = TextMetrics.levenshtein_similarity("cat", "dog")
        self.assertLess(result.score, 0.5)
    
    def test_coherence_with_transitions(self):
        """Text with transitions should score higher"""
        text = """First, we discuss the problem. Therefore, we need solutions.
        Furthermore, we must consider alternatives. Finally, we conclude."""
        result = TextMetrics.coherence_score(text)
        self.assertGreater(result.score, 0.3)
    
    def test_factual_density_with_numbers(self):
        """Text with numbers should have higher density"""
        text = "In 2024, the company had 500 employees and $10 million revenue."
        result = TextMetrics.factual_density(text)
        self.assertGreater(result.score, 0.2)
        self.assertGreater(result.details["numbers"], 0)


class TestMultiMetricEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.evaluator = MultiMetricEvaluator()
    
    def test_evaluate_without_reference(self):
        """Evaluate answer without reference"""
        answer = "This is a coherent answer with proper structure. It contains facts and details."
        result = self.evaluator.evaluate(answer)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.metrics), 0)
        self.assertIn(result.quality_tier, ["excellent", "good", "fair", "poor"])
    
    def test_evaluate_with_reference(self):
        """Evaluate answer with reference"""
        answer = "The sky is blue due to Rayleigh scattering of sunlight."
        reference = "The sky appears blue because of Rayleigh scattering."
        
        result = self.evaluator.evaluate(answer, reference)
        
        self.assertIsNotNone(result)
        # Should have comparison metrics
        metric_names = [m.name for m in result.metrics]
        self.assertIn("bleu", metric_names)
        self.assertIn("rouge-1", metric_names)
    
    def test_quality_tiers(self):
        """Test quality tier assignment"""
        # High quality answer
        good_answer = """First, let me explain the concept clearly.
        The key point is that this process involves multiple steps.
        Furthermore, we should consider the implications.
        In conclusion, the answer is well-supported by evidence."""
        
        result = self.evaluator.evaluate(good_answer)
        self.assertIn(result.quality_tier, ["excellent", "good", "fair"])


class TestCompareAnswers(unittest.TestCase):
    
    def test_compare_multiple_answers(self):
        """Compare multiple answers"""
        answers = [
            "Short answer",
            "A more detailed answer with proper explanation and coherent structure.",
            "Another answer with some details"
        ]
        model_ids = ["model_a", "model_b", "model_c"]
        
        result = compare_answers(answers, model_ids)
        
        self.assertIn("rankings", result)
        self.assertEqual(len(result["rankings"]), 3)
        self.assertEqual(result["rankings"][0]["rank"], 1)
    
    def test_compare_with_reference(self):
        """Compare answers against reference"""
        reference = "The sky is blue due to Rayleigh scattering."
        answers = [
            "The sky is blue because of scattering.",
            "The ocean is blue.",
            "Rayleigh scattering causes the blue sky color."
        ]
        model_ids = ["a", "b", "c"]
        
        result = compare_answers(answers, model_ids, reference)
        
        # Answer closest to reference should rank higher
        self.assertIsNotNone(result["best_answer"])


class TestConvenienceFunctions(unittest.TestCase):
    
    def test_evaluate_answer_function(self):
        """Test convenience function"""
        result = evaluate_answer("This is a test answer.")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.aggregate_score)


if __name__ == '__main__':
    unittest.main()
