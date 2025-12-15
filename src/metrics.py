"""
Multi-Metric Evaluation Module
Level 2 Upgrade: Comprehensive answer quality assessment

Evaluates answers using multiple metrics:
- BLEU (n-gram precision)
- ROUGE (recall-oriented)
- Levenshtein distance
- Semantic coherence
- Factual density
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric evaluation"""
    name: str
    score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiMetricEvaluation:
    """Complete multi-metric evaluation result"""
    answer: str
    reference: Optional[str]
    metrics: List[MetricResult]
    aggregate_score: float
    quality_tier: str  # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]


class TextMetrics:
    """Collection of text quality metrics"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple word tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    @staticmethod
    def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # ========================================
    # BLEU Score
    # ========================================
    @classmethod
    def bleu_score(
        cls,
        candidate: str,
        reference: str,
        max_n: int = 4
    ) -> MetricResult:
        """
        Calculate BLEU score (simplified version).
        
        Measures n-gram precision of candidate against reference.
        """
        cand_tokens = cls.tokenize(candidate)
        ref_tokens = cls.tokenize(reference)
        
        if not cand_tokens or not ref_tokens:
            return MetricResult(name="bleu", score=0.0, details={"error": "empty input"})
        
        precisions = []
        
        for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
            cand_ngrams = cls.get_ngrams(cand_tokens, n)
            ref_ngrams = cls.get_ngrams(ref_tokens, n)
            
            if not cand_ngrams:
                continue
            
            ref_counts = Counter(ref_ngrams)
            cand_counts = Counter(cand_ngrams)
            
            # Count matches
            matches = 0
            for ngram, count in cand_counts.items():
                matches += min(count, ref_counts.get(ngram, 0))
            
            precision = matches / len(cand_ngrams) if cand_ngrams else 0
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            return MetricResult(name="bleu", score=0.0, details={"precisions": precisions})
        
        # Geometric mean of precisions
        log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
        avg_log = sum(log_precisions) / len(log_precisions)
        
        if avg_log == -float('inf'):
            bleu = 0.0
        else:
            bleu = math.exp(avg_log)
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) > 0 else 0
        
        final_score = bp * bleu
        
        return MetricResult(
            name="bleu",
            score=final_score,
            details={
                "precisions": precisions,
                "brevity_penalty": bp,
                "candidate_length": len(cand_tokens),
                "reference_length": len(ref_tokens)
            }
        )
    
    # ========================================
    # ROUGE Score
    # ========================================
    @classmethod
    def rouge_score(
        cls,
        candidate: str,
        reference: str,
        rouge_type: str = "rouge-l"
    ) -> MetricResult:
        """
        Calculate ROUGE score.
        
        ROUGE-1: Unigram recall
        ROUGE-2: Bigram recall
        ROUGE-L: Longest common subsequence
        """
        cand_tokens = cls.tokenize(candidate)
        ref_tokens = cls.tokenize(reference)
        
        if not ref_tokens:
            return MetricResult(name=rouge_type, score=0.0, details={"error": "empty reference"})
        
        if rouge_type == "rouge-1":
            # Unigram recall
            cand_set = set(cand_tokens)
            ref_set = set(ref_tokens)
            overlap = len(cand_set & ref_set)
            recall = overlap / len(ref_set) if ref_set else 0
            precision = overlap / len(cand_set) if cand_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return MetricResult(
                name="rouge-1",
                score=f1,
                details={"precision": precision, "recall": recall, "f1": f1}
            )
        
        elif rouge_type == "rouge-2":
            # Bigram recall
            cand_bigrams = set(cls.get_ngrams(cand_tokens, 2))
            ref_bigrams = set(cls.get_ngrams(ref_tokens, 2))
            
            if not ref_bigrams:
                return MetricResult(name="rouge-2", score=0.0, details={"error": "no bigrams in reference"})
            
            overlap = len(cand_bigrams & ref_bigrams)
            recall = overlap / len(ref_bigrams)
            precision = overlap / len(cand_bigrams) if cand_bigrams else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return MetricResult(
                name="rouge-2",
                score=f1,
                details={"precision": precision, "recall": recall, "f1": f1}
            )
        
        else:  # rouge-l
            # Longest Common Subsequence
            lcs_length = cls._lcs_length(cand_tokens, ref_tokens)
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0
            precision = lcs_length / len(cand_tokens) if cand_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return MetricResult(
                name="rouge-l",
                score=f1,
                details={"lcs_length": lcs_length, "precision": precision, "recall": recall, "f1": f1}
            )
    
    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(x), len(y)
        
        # Optimization for large sequences
        if m > 500 or n > 500:
            # Use approximate method for very long sequences
            x_set, y_set = set(x), set(y)
            return len(x_set & y_set)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # ========================================
    # Levenshtein Distance
    # ========================================
    @classmethod
    def levenshtein_similarity(cls, text1: str, text2: str) -> MetricResult:
        """
        Calculate normalized Levenshtein similarity.
        
        1.0 = identical, 0.0 = completely different
        """
        tokens1 = cls.tokenize(text1)
        tokens2 = cls.tokenize(text2)
        
        if not tokens1 and not tokens2:
            return MetricResult(name="levenshtein", score=1.0, details={"distance": 0})
        
        if not tokens1 or not tokens2:
            return MetricResult(name="levenshtein", score=0.0, details={"distance": max(len(tokens1), len(tokens2))})
        
        # Use word-level edit distance
        m, n = len(tokens1), len(tokens2)
        
        # Optimization for very long sequences
        if m > 200 or n > 200:
            # Use character-level sampling
            text1_sample = ' '.join(tokens1[:100])
            text2_sample = ' '.join(tokens2[:100])
            m, n = len(text1_sample), len(text2_sample)
            
            if m > n:
                text1_sample, text2_sample = text2_sample, text1_sample
                m, n = n, m
            
            previous_row = list(range(n + 1))
            for i, c1 in enumerate(text1_sample, 1):
                current_row = [i]
                for j, c2 in enumerate(text2_sample, 1):
                    insertions = previous_row[j] + 1
                    deletions = current_row[j-1] + 1
                    substitutions = previous_row[j-1] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            distance = previous_row[-1]
            max_len = max(len(text1_sample), len(text2_sample))
        else:
            # Word-level edit distance
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if tokens1[i-1] == tokens2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            distance = dp[m][n]
            max_len = max(m, n)
        
        similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
        
        return MetricResult(
            name="levenshtein",
            score=similarity,
            details={"distance": distance, "max_length": max_len}
        )
    
    # ========================================
    # Coherence Score
    # ========================================
    @classmethod
    def coherence_score(cls, text: str) -> MetricResult:
        """
        Estimate text coherence based on structural features.
        
        Measures:
        - Sentence connectivity
        - Paragraph structure
        - Transition word usage
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return MetricResult(
                name="coherence",
                score=0.5,
                details={"note": "too few sentences to evaluate"}
            )
        
        # Transition words that indicate coherence
        transitions = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'thus', 'hence', 'meanwhile', 'nevertheless',
            'first', 'second', 'third', 'finally', 'next', 'then',
            'for example', 'for instance', 'specifically', 'in particular',
            'in conclusion', 'in summary', 'overall', 'to summarize'
        }
        
        text_lower = text.lower()
        transition_count = sum(1 for t in transitions if t in text_lower)
        
        # Sentence length variance (lower is more coherent)
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_score = 1 / (1 + variance / 100)  # Normalize
        
        # Pronoun continuity (indicates referential coherence)
        pronouns = {'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those'}
        pronoun_count = sum(1 for w in text_lower.split() if w in pronouns)
        pronoun_score = min(1.0, pronoun_count / (len(sentences) * 2))
        
        # Combine scores
        transition_score = min(1.0, transition_count / (len(sentences) / 2))
        
        final_score = (length_score * 0.3 + pronoun_score * 0.3 + transition_score * 0.4)
        
        return MetricResult(
            name="coherence",
            score=final_score,
            details={
                "sentence_count": len(sentences),
                "transition_count": transition_count,
                "avg_sentence_length": avg_length,
                "length_variance": variance
            }
        )
    
    # ========================================
    # Factual Density
    # ========================================
    @classmethod
    def factual_density(cls, text: str) -> MetricResult:
        """
        Estimate factual density of text.
        
        Measures presence of:
        - Numbers and quantities
        - Proper nouns (capitalized words)
        - Technical terms
        - Specific references
        """
        tokens = text.split()
        
        if not tokens:
            return MetricResult(name="factual_density", score=0.0, details={"error": "empty text"})
        
        # Count numbers
        numbers = len(re.findall(r'\d+(?:\.\d+)?%?', text))
        
        # Count capitalized words (potential proper nouns), excluding sentence starts
        sentences = re.split(r'[.!?]\s+', text)
        proper_nouns = 0
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i > 0 and word and word[0].isupper():
                    proper_nouns += 1
        
        # Count specific patterns (dates, measurements, etc.)
        specifics = len(re.findall(
            r'\b(?:\d{4}|\d{1,2}/\d{1,2}|\d+\s*(?:kg|km|m|cm|%|dollars?|euros?))\b',
            text, re.IGNORECASE
        ))
        
        # Technical indicators
        technical_patterns = r'\b(?:algorithm|function|process|system|method|analysis|research|study|data|results?)\b'
        technical = len(re.findall(technical_patterns, text, re.IGNORECASE))
        
        total_indicators = numbers + proper_nouns + specifics + technical
        density = total_indicators / len(tokens)
        
        # Normalize to 0-1 scale (typical density is 0.05-0.20)
        normalized_score = min(1.0, density / 0.15)
        
        return MetricResult(
            name="factual_density",
            score=normalized_score,
            details={
                "numbers": numbers,
                "proper_nouns": proper_nouns,
                "specifics": specifics,
                "technical_terms": technical,
                "total_indicators": total_indicators,
                "token_count": len(tokens),
                "raw_density": density
            }
        )


class MultiMetricEvaluator:
    """
    Comprehensive answer evaluator using multiple metrics.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Custom weights for each metric (default: equal weights)
        """
        self.default_weights = {
            "bleu": 0.15,
            "rouge-1": 0.15,
            "rouge-l": 0.15,
            "levenshtein": 0.10,
            "coherence": 0.25,
            "factual_density": 0.20
        }
        self.weights = weights or self.default_weights
    
    def evaluate(
        self,
        answer: str,
        reference: Optional[str] = None
    ) -> MultiMetricEvaluation:
        """
        Evaluate an answer using all available metrics.
        
        Args:
            answer: The answer to evaluate
            reference: Optional reference answer for comparison metrics
            
        Returns:
            MultiMetricEvaluation with all metric results
        """
        metrics = []
        
        # Intrinsic metrics (no reference needed)
        metrics.append(TextMetrics.coherence_score(answer))
        metrics.append(TextMetrics.factual_density(answer))
        
        # Comparison metrics (if reference provided)
        if reference:
            metrics.append(TextMetrics.bleu_score(answer, reference))
            metrics.append(TextMetrics.rouge_score(answer, reference, "rouge-1"))
            metrics.append(TextMetrics.rouge_score(answer, reference, "rouge-l"))
            metrics.append(TextMetrics.levenshtein_similarity(answer, reference))
        
        # Calculate aggregate score
        total_weight = 0
        weighted_sum = 0
        
        for metric in metrics:
            weight = self.weights.get(metric.name, 0.1)
            weighted_sum += metric.score * weight
            total_weight += weight
        
        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine quality tier
        if aggregate_score >= 0.8:
            quality_tier = "excellent"
        elif aggregate_score >= 0.6:
            quality_tier = "good"
        elif aggregate_score >= 0.4:
            quality_tier = "fair"
        else:
            quality_tier = "poor"
        
        # Generate recommendations
        recommendations = []
        for metric in metrics:
            if metric.score < 0.5:
                if metric.name == "coherence":
                    recommendations.append("Improve text structure with transition words and consistent sentence lengths")
                elif metric.name == "factual_density":
                    recommendations.append("Add more specific facts, numbers, or references")
                elif "rouge" in metric.name or metric.name == "bleu":
                    recommendations.append("Answer diverges significantly from reference - consider revising key points")
                elif metric.name == "levenshtein":
                    recommendations.append("Answer structure differs from reference - verify content accuracy")
        
        return MultiMetricEvaluation(
            answer=answer[:200] + "..." if len(answer) > 200 else answer,
            reference=reference[:100] + "..." if reference and len(reference) > 100 else reference,
            metrics=metrics,
            aggregate_score=aggregate_score,
            quality_tier=quality_tier,
            recommendations=recommendations
        )
    
    def compare_answers(
        self,
        answers: List[str],
        model_ids: List[str],
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple answers and rank them by quality.
        
        Returns ranking and detailed comparison.
        """
        evaluations = []
        
        for i, answer in enumerate(answers):
            eval_result = self.evaluate(answer, reference)
            evaluations.append({
                "model_id": model_ids[i] if i < len(model_ids) else f"model_{i}",
                "evaluation": eval_result,
                "aggregate_score": eval_result.aggregate_score
            })
        
        # Sort by aggregate score
        evaluations.sort(key=lambda x: x["aggregate_score"], reverse=True)
        
        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "model_id": e["model_id"],
                    "score": e["aggregate_score"],
                    "tier": e["evaluation"].quality_tier
                }
                for i, e in enumerate(evaluations)
            ],
            "detailed_evaluations": evaluations,
            "best_answer": evaluations[0] if evaluations else None,
            "score_spread": max(e["aggregate_score"] for e in evaluations) - min(e["aggregate_score"] for e in evaluations) if len(evaluations) > 1 else 0
        }


# Singleton instance
evaluator = MultiMetricEvaluator()


def evaluate_answer(answer: str, reference: Optional[str] = None) -> MultiMetricEvaluation:
    """Convenience function for quick evaluation"""
    return evaluator.evaluate(answer, reference)


def compare_answers(
    answers: List[str],
    model_ids: List[str],
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for comparing multiple answers"""
    return evaluator.compare_answers(answers, model_ids, reference)
