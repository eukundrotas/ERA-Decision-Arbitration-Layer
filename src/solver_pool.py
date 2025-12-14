import json
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models import SolverOutput
from src.providers import provider
from src.prompts import get_domain_prompts
from src.schemas import validate_solver_json
from src.utils import safe_parse_json

logger = logging.getLogger(__name__)

class SolverSpec:
    """Спецификация одного solver'а"""
    def __init__(self, model_id: str, temperature: float = 0.7):
        self.model_id = model_id
        self.temperature = temperature

class SolverPool:
    """Пул solver-моделей"""
    
    # Стандартные пулы по доменам
    POOLS = {
        "science": [
            SolverSpec("openai/gpt-4-turbo-preview", 0.7),
            SolverSpec("anthropic/claude-3-opus", 0.7),
            SolverSpec("meta-llama/llama-3-70b-instruct", 0.6),
            SolverSpec("mistralai/mistral-large", 0.7),
            SolverSpec("deepseek/deepseek-chat", 0.6),
            SolverSpec("google/gemini-2.0-flash-exp", 0.7),
            SolverSpec("x-ai/grok-2", 0.7),
        ],
        "math": [
            SolverSpec("openai/gpt-4-turbo-preview", 0.3),
            SolverSpec("anthropic/claude-3-opus", 0.3),
            SolverSpec("deepseek/deepseek-reasoner", 0.2),
            SolverSpec("meta-llama/llama-3-70b-instruct", 0.5),
            SolverSpec("mistralai/mistral-large", 0.4),
            SolverSpec("google/gemini-2.0-flash-exp", 0.3),
        ],
        "med": [
            SolverSpec("anthropic/claude-3-opus", 0.5),
            SolverSpec("openai/gpt-4-turbo-preview", 0.5),
            SolverSpec("mistralai/mistral-large", 0.5),
            SolverSpec("google/gemini-2.0-flash-exp", 0.5),
            SolverSpec("meta-llama/llama-3-70b-instruct", 0.5),
        ],
        "econ": [
            SolverSpec("openai/gpt-4-turbo-preview", 0.6),
            SolverSpec("anthropic/claude-3-opus", 0.6),
            SolverSpec("mistralai/mistral-large", 0.6),
            SolverSpec("google/gemini-2.0-flash-exp", 0.6),
            SolverSpec("meta-llama/llama-3-70b-instruct", 0.6),
            SolverSpec("deepseek/deepseek-chat", 0.6),
        ]
    }
    
    def __init__(self, pool_name: str = "science"):
        self.pool_name = pool_name
        self.solvers = self.POOLS.get(pool_name, self.POOLS["science"])
        logger.info(f"Initialized pool {pool_name} with {len(self.solvers)} solvers")
    
    def run_parallel(
        self,
        problem: str,
        task_id: str,
        domain: str = "science",
        max_workers: int = 8
    ) -> List[SolverOutput]:
        """Запускает solvers параллельно"""
        
        domain_prompts = get_domain_prompts(domain)
        solver_system = domain_prompts["solver_system"]
        solver_user_template = """Task ID: {task_id}

Задача:
{problem}

Реши эту задачу, следуя инструкциям в системном промпте."""
        
        results = []
        
        def call_solver(spec: SolverSpec):
            try:
                start = time.time()
                user_msg = solver_user_template.format(task_id=task_id, problem=problem)
                raw = provider.generate(
                    system_prompt=solver_system,
                    user_prompt=user_msg,
                    model_id=spec.model_id,
                    temperature=spec.temperature
                )
                latency_ms = int((time.time() - start) * 1000)
                
                parsed = safe_parse_json(raw)
                validate_solver_json(parsed)
                
                return SolverOutput(
                    model_id=spec.model_id,
                    task_id=task_id,
                    final_answer=parsed["final_answer"],
                    confidence=float(parsed["confidence"]),
                    assumptions=list(parsed["assumptions"]),
                    risks=list(parsed["risks"]),
                    evidence=list(parsed["evidence"]),
                    self_checks=list(parsed["self_checks"]),
                    latency_ms=latency_ms
                )
            except Exception as e:
                logger.error(f"Solver {spec.model_id} failed: {e}")
                raise
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(self.solvers))) as executor:
            futures = [executor.submit(call_solver, spec) for spec in self.solvers]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        return results
