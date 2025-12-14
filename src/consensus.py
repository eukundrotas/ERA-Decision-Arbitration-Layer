import json
import logging
from typing import List, Dict, Any
from src.models import SolverOutput, ArbiterOutput
from src.providers import provider
from src.prompts import get_domain_prompts
from src.utils import safe_parse_json

logger = logging.getLogger(__name__)

class ConsensusSynthesizer:
    """Синтезирует консенсус из top-K ответов"""
    
    CONSENSUS_MODELS = {
        "science": "openai/gpt-4-turbo-preview",
        "math": "openai/gpt-4-turbo-preview",
        "med": "anthropic/claude-3-opus",
        "econ": "openai/gpt-4-turbo-preview"
    }
    
    def __init__(self, domain: str = "science"):
        self.domain = domain
        self.model_id = self.CONSENSUS_MODELS.get(domain, "openai/gpt-4-turbo-preview")
    
    def synthesize(
        self,
        problem: str,
        task_id: str,
        top_candidates: List[SolverOutput],
        topk: int = 3
    ) -> Dict[str, Any]:
        """Синтезирует финальный ответ из top-K"""
        
        domain_prompts = get_domain_prompts(self.domain)
        consensus_system = domain_prompts["consensus_system"].replace("{topk}", str(topk))
        
        top_answers_json = json.dumps(
            [c.to_dict() for c in top_candidates[:topk]],
            ensure_ascii=False,
            indent=2
        )
        
        user_msg = f"""Task ID: {task_id}

Исходная задача:
{problem}

Топ {topk} ответов:
{top_answers_json}

Синтезируй финальный ответ, сохраняя правильные части и разрешая противоречия явно."""
        
        try:
            raw = provider.generate(
                system_prompt=consensus_system,
                user_prompt=user_msg,
                model_id=self.model_id,
                temperature=0.5
            )
            
            parsed = safe_parse_json(raw)
            
            return {
                "final_answer": parsed.get("final_answer", ""),
                "synthesis_notes": parsed.get("synthesis_notes", []),
                "sources": [c.model_id for c in top_candidates[:topk]]
            }
        except Exception as e:
            logger.error(f"Consensus synthesis failed: {e}")
            raise
