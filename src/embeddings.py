"""
Semantic Similarity Clustering Module
Level 1 Upgrade: Enhanced disagreement detection through embeddings

Uses sentence embeddings to cluster similar answers and detect
real disagreements vs superficial wording differences.
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Result of semantic clustering"""
    cluster_id: int
    answers: List[str]
    model_ids: List[str]
    centroid_answer: str
    semantic_similarity: float

@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement between solver responses"""
    num_clusters: int
    clusters: List[ClusterResult]
    disagreement_score: float  # 0.0 = full agreement, 1.0 = full disagreement
    dominant_cluster_size: int
    is_significant_disagreement: bool
    recommendation: str  # 'consensus', 'hard_select', 'rebuttal'


class SemanticClusterer:
    """
    Clusters solver answers by semantic similarity.
    Uses a lightweight approach with text hashing and Jaccard similarity
    for environments without heavy ML dependencies.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Args:
            similarity_threshold: Minimum similarity to group answers (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def _tokenize(self, text: str) -> set:
        """Simple word tokenization with normalization"""
        # Lowercase, remove punctuation, split by whitespace
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = set(text.split())
        # Also add bigrams for better matching
        word_list = text.split()
        bigrams = set()
        for i in range(len(word_list) - 1):
            bigrams.add(f"{word_list[i]}_{word_list[i+1]}")
        return words.union(bigrams)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_similarity_matrix(self, answers: List[str]) -> List[List[float]]:
        """Compute pairwise similarity matrix"""
        n = len(answers)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self._jaccard_similarity(answers[i], answers[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        
        return matrix
    
    def cluster(
        self,
        answers: List[str],
        model_ids: List[str]
    ) -> DisagreementAnalysis:
        """
        Cluster answers by semantic similarity.
        
        Args:
            answers: List of answer strings from solvers
            model_ids: Corresponding model IDs
            
        Returns:
            DisagreementAnalysis with clustering results
        """
        if not answers:
            return DisagreementAnalysis(
                num_clusters=0,
                clusters=[],
                disagreement_score=0.0,
                dominant_cluster_size=0,
                is_significant_disagreement=False,
                recommendation='hard_select'
            )
        
        n = len(answers)
        
        # Single answer case
        if n == 1:
            return DisagreementAnalysis(
                num_clusters=1,
                clusters=[ClusterResult(
                    cluster_id=0,
                    answers=answers,
                    model_ids=model_ids,
                    centroid_answer=answers[0],
                    semantic_similarity=1.0
                )],
                disagreement_score=0.0,
                dominant_cluster_size=1,
                is_significant_disagreement=False,
                recommendation='hard_select'
            )
        
        # Compute similarity matrix
        sim_matrix = self._compute_similarity_matrix(answers)
        
        # Simple agglomerative clustering
        clusters = self._agglomerative_cluster(answers, model_ids, sim_matrix)
        
        # Calculate disagreement score
        dominant_size = max(len(c.answers) for c in clusters) if clusters else 0
        disagreement_score = 1.0 - (dominant_size / n)
        
        # Determine recommendation
        is_significant = len(clusters) > 1 and disagreement_score > 0.3
        
        if disagreement_score < 0.2:
            recommendation = 'hard_select'
        elif disagreement_score < 0.5:
            recommendation = 'consensus'
        else:
            recommendation = 'rebuttal'
        
        return DisagreementAnalysis(
            num_clusters=len(clusters),
            clusters=clusters,
            disagreement_score=disagreement_score,
            dominant_cluster_size=dominant_size,
            is_significant_disagreement=is_significant,
            recommendation=recommendation
        )
    
    def _agglomerative_cluster(
        self,
        answers: List[str],
        model_ids: List[str],
        sim_matrix: List[List[float]]
    ) -> List[ClusterResult]:
        """Simple agglomerative clustering"""
        n = len(answers)
        
        # Each answer starts in its own cluster
        cluster_assignments = list(range(n))
        
        # Merge clusters if similarity >= threshold
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.similarity_threshold:
                    # Merge cluster j into cluster i
                    old_cluster = cluster_assignments[j]
                    new_cluster = cluster_assignments[i]
                    for k in range(n):
                        if cluster_assignments[k] == old_cluster:
                            cluster_assignments[k] = new_cluster
        
        # Group by cluster assignment
        cluster_groups: Dict[int, List[int]] = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(idx)
        
        # Build ClusterResult objects
        results = []
        for new_id, (_, indices) in enumerate(cluster_groups.items()):
            cluster_answers = [answers[i] for i in indices]
            cluster_models = [model_ids[i] for i in indices]
            
            # Find centroid (answer most similar to all others in cluster)
            if len(cluster_answers) == 1:
                centroid = cluster_answers[0]
                avg_sim = 1.0
            else:
                best_centroid_idx = 0
                best_avg_sim = 0.0
                for i, idx_i in enumerate(indices):
                    avg_sim = sum(sim_matrix[idx_i][idx_j] for idx_j in indices) / len(indices)
                    if avg_sim > best_avg_sim:
                        best_avg_sim = avg_sim
                        best_centroid_idx = i
                centroid = cluster_answers[best_centroid_idx]
                avg_sim = best_avg_sim
            
            results.append(ClusterResult(
                cluster_id=new_id,
                answers=cluster_answers,
                model_ids=cluster_models,
                centroid_answer=centroid,
                semantic_similarity=avg_sim
            ))
        
        # Sort by cluster size (largest first)
        results.sort(key=lambda x: len(x.answers), reverse=True)
        
        return results


# Singleton instance
clusterer = SemanticClusterer()


def analyze_disagreement(
    answers: List[str],
    model_ids: List[str],
    threshold: float = 0.6
) -> DisagreementAnalysis:
    """
    Convenience function to analyze disagreement in solver responses.
    
    Args:
        answers: List of answer strings
        model_ids: Corresponding model IDs
        threshold: Similarity threshold for clustering
        
    Returns:
        DisagreementAnalysis with recommendations
    """
    clusterer.similarity_threshold = threshold
    return clusterer.cluster(answers, model_ids)
