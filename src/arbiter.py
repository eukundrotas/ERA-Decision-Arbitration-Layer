import json
import logging
from typing import List, Dict, Any
from src.models import SolverOutput, ArbiterOutput, ArbiterRanking
from src.providers import provider
from src.prompts import get_domain_prompts
from src.schemas import validate_arbiter_json
from src.utils import safe_parse_json

logger = logging.getLogger(__name__)

class Arbiter:
    """Модель-арбитр"""
    
    ARBITER_MODELS = {
        "science": "openai/gpt-4-turbo-preview",
        "math": "openai/gpt-4-turbo-preview",
        "med": "anthropic/claude-3-opus",
        "econ": "openai/gpt-4-turbo-preview"
    }
    
    def __init__(self, domain: str = "science"):
        self.domain = domain
        self.model_id = self.ARBITER_MODELS.get(domain, "openai/gpt-4-turbo-preview")
    
    def rank(
        self,
        problem: str,
        task_id: str,
        candidates: List[SolverOutput]
    ) -> ArbiterOutput:
        """Ранжирует solver-ответы"""
        
        domain_prompts = get_domain_prompts(self.domain)
        arbiter_system = domain_prompts["arbiter_system"]
        
        candidates_json = json.dumps(
            [c.to_dict() for c in candidates],
            ensure_ascii=False,
            indent=2
        )
        
        user_msg = f"""Task ID: {task_id}

Исходная задача:
{problem}

Ответы solvers:
{candidates_json}

Оцени каждый ответ по критериям качества и выбери лучший."""
        
        try:
            raw = provider.generate(
                system_prompt=arbiter_system,
                user_prompt=user_msg,
                model_id=self.model_id,
                temperature=0.3
            )
            
            parsed = safe_parse_json(raw)
            validate_arbiter_json(parsed)
            
            ranking = [
                ArbiterRanking(
                    model_id=r["model_id"],
                    score=float(r["score"])
                )
                for r in parsed["ranking"]
            ]
            ranking.sort(key=lambda x: x.score, reverse=True)
            
            return ArbiterOutput(
                task_id=task_id,
                selected_model_id=parsed["selected_model_id"],
                ranking=ranking,
                final_answer=parsed["final_answer"],
                arbiter_notes=list(parsed["arbiter_notes"]),
                decision_mode=parsed.get("decision_mode", "hard_select"),
                used_candidates=[parsed["selected_model_id"]]
            )
        except Exception as e:
            logger.error(f"Arbiter failed: {e}")
            raise
