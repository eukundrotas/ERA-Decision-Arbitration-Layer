import logging
from typing import List, Dict, Any
from datetime import datetime
from src.models import SolverOutput, ArbiterOutput, RunRecord
from src.solver_pool import SolverPool
from src.arbiter import Arbiter
from src.consensus import ConsensusSynthesizer
from src.rebuttal import RebuttalRound
from src.stability import StabilityAnalyzer
from src.model_memory import ModelMemory
from src.utils import write_run_record, write_final_json, write_model_quality

logger = logging.getLogger(__name__)

class MultiLLMOrchestrator:
    """Главный оркестратор системы"""
    
    def __init__(
        self,
        pool_name: str = "science",
        domain: str = "science",
        out_dir: str = "./out"
    ):
        self.pool = SolverPool(pool_name)
        self.arbiter = Arbiter(domain)
        self.consensus = ConsensusSynthesizer(domain)
        self.rebuttal = RebuttalRound(domain)
        self.stability_analyzer = StabilityAnalyzer()
        self.model_memory = ModelMemory(f"{out_dir}/model_quality.json")
        self.domain = domain
        self.out_dir = out_dir
    
    def run_single(
        self,
        problem: str,
        task_id: str,
        consensus_topk: int = 3,
        epsilon: float = 0.07,
        enable_rebuttal: bool = True,
        disagreement_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """Запускает один цикл (одна задача, один прогон)"""
        
        logger.info(f"=== Single Run: {task_id} ===")
        
        # Раунд 1: Solvers
        logger.info("Round 1: Solver Pool")
        candidates_r1 = self.pool.run_parallel(
            problem=problem,
            task_id=task_id,
            domain=self.domain
        )
        logger.info(f"Got {len(candidates_r1)} solver outputs")
        
        # Disagreement check
        unique_answers = len(set(c.final_answer for c in candidates_r1))
        disagreement_rate = unique_answers / len(candidates_r1)
        logger.info(f"Disagreement rate: {disagreement_rate:.2f}")
        
        # Раунд 2: Arbiter (первый)
        logger.info("Round 2: Arbiter (R1)")
        arbiter_out = self.arbiter.rank(problem, task_id, candidates_r1)
        logger.info(f"Arbiter selected: {arbiter_out.selected_model_id} (score={arbiter_out.ranking[0].score:.2f})")
        
        candidates_current = candidates_r1
        arbiter_current = arbiter_out
        round_number = 1
        
        # Rebuttal (если разброс высокий)
        if enable_rebuttal and disagreement_rate >= disagreement_threshold:
            logger.info("Rebuttal Round enabled")
            candidates_r2 = self.rebuttal.run(problem, task_id, candidates_current)
            
            # Пересчитаем disagreement
            unique_r2 = len(set(c.final_answer for c in candidates_r2))
            disagreement_rate = unique_r2 / len(candidates_r2)
            logger.info(f"After rebuttal, disagreement: {disagreement_rate:.2f}")
            
            # Повторный арбитраж
            arbiter_current = self.arbiter.rank(problem, task_id, candidates_r2)
            candidates_current = candidates_r2
            round_number = 2
        
        # Consensus check
        decision_mode = "hard_select"
        used_candidates = [arbiter_current.selected_model_id]
        final_answer = arbiter_current.final_answer
        
        if len(arbiter_current.ranking) >= 2:
            gap = arbiter_current.ranking[0].score - arbiter_current.ranking[1].score
            if gap < epsilon:
                logger.info(f"Gap {gap:.3f} < epsilon {epsilon}, trying consensus")
                try:
                    top_k = min(consensus_topk, len(candidates_current))
                    top_models = [c.model_id for c in candidates_current if c.model_id in [r.model_id for r in arbiter_current.ranking[:top_k]]]
                    top_candidates = [c for c in candidates_current if c.model_id in top_models]
                    
                    consensus_result = self.consensus.synthesize(
                        problem,
                        task_id,
                        top_candidates,
                        topk=top_k
                    )
                    
                    decision_mode = f"consensus_top{top_k}"
                    used_candidates = consensus_result["sources"]
                    final_answer = consensus_result["final_answer"]
                    logger.info(f"Consensus: {decision_mode}, sources={used_candidates}")
                except Exception as e:
                    logger.warning(f"Consensus failed: {e}, falling back to hard_select")
        
        # Обновляем arbiter_current с финальным ответом
        arbiter_current.final_answer = final_answer
        arbiter_current.decision_mode = decision_mode
        arbiter_current.used_candidates = used_candidates
        
        # Обновляем model memory
        for model_id in used_candidates:
            self.model_memory.update_reliability(model_id, reward=1.0)
        for c in candidates_current:
            if c.model_id not in used_candidates:
                self.model_memory.update_reliability(c.model_id, reward=0.0)
        
        # Записываем runs
        for i, candidate in enumerate(candidates_current):
            record = RunRecord(
                pool=self.pool.pool_name,
                iter_index=0,  # будет заполняться выше
                round_number=round_number,
                solver_model_id=candidate.model_id,
                latency_ms=candidate.latency_ms,
                confidence=candidate.confidence,
                arbiter_score=next((r.score for r in arbiter_current.ranking if r.model_id == candidate.model_id), None),
                decision_mode=decision_mode,
                used_candidates=",".join(used_candidates),
                final_answer=final_answer[:100] + "..." if len(final_answer) > 100 else final_answer,
                notes="",
                timestamp=datetime.utcnow().isoformat()
            )
            write_run_record(record, self.out_dir)
        
        return {
            "task_id": task_id,
            "arbiter_output": arbiter_current.to_dict(),
            "decision_mode": decision_mode,
            "used_candidates": used_candidates,
            "final_answer": final_answer,
            "disagreement_rate": disagreement_rate
        }
    
    def run_multi(
        self,
        problem: str,
        task_id: str,
        repeats: int = 5,
        consensus_topk: int = 3,
        epsilon: float = 0.07,
        enable_rebuttal: bool = True
    ) -> Dict[str, Any]:
        """Запускает multi-run для оценки stability"""
        
        logger.info(f"=== Multi-Run: {task_id} ({repeats} repeats) ===")
        
        results_per_run = []
        
        for i in range(repeats):
            logger.info(f"--- Run {i+1}/{repeats} ---")
            result = self.run_single(
                problem,
                task_id,
                consensus_topk=consensus_topk,
                epsilon=epsilon,
                enable_rebuttal=enable_rebuttal
            )
            results_per_run.append({
                "decision_mode": result["decision_mode"],
                "used_candidates": result["used_candidates"],
                "final_answer": result["final_answer"]
            })
        
        # Stability analysis
        stability = self.stability_analyzer.analyze(results_per_run)
        
        final_result = results_per_run[-1]  # последний результат как финальный
        
        final_data = {
            "task_id": task_id,
            "final_answer": final_result["final_answer"],
            "decision_mode": final_result["decision_mode"],
            "used_candidates": final_result["used_candidates"],
            "stability": stability,
            "config": {
                "pool": self.pool.pool_name,
                "domain": self.domain,
                "repeats": repeats,
                "consensus_topk": consensus_topk,
                "epsilon": epsilon,
                "rebuttal": enable_rebuttal
            }
        }
        
        write_final_json(final_data, self.out_dir)
        self.model_memory.save()
        
        logger.info(f"Stability: {stability}")
        logger.info(f"Final answer saved to {self.out_dir}/final.json")
        
        return final_data
