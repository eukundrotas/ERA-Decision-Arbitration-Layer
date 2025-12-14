import json
import logging
from typing import List, Dict, Any
from src.models import SolverOutput
from src.providers import provider
from src.prompts import get_domain_prompts
from src.utils import safe_parse_json
from src.schemas import validate_solver_json

logger = logging.getLogger(__name__)

class RebuttalRound:
    """Раунд критики и улучшения"""
    
    def __init__(self, domain: str = "science"):
        self.domain = domain
    
    def run(
        self,
        problem: str,
        task_id: str,
        all_outputs: List[SolverOutput]
    ) -> List[SolverOutput]:
        """Запускает rebuttal-раунд"""
        
        domain_prompts = get_domain_prompts(self.domain)
        rebuttal_system = domain_prompts["rebuttal_system"]
        
        improved = []
        
        for own_output in all_outputs:
            other_outputs = [o for o in all_outputs if o.model_id != own_output.model_id]
            
            other_json = json.dumps(
                [o.to_dict() for o in other_outputs],
                ensure_ascii=False,
                indent=2
            )
            
            user_msg = f"""Task ID: {task_id}

Исходная задача:
{problem}

Другие ответы (раунд 1):
{other_json}

Твой ответ раунда 1:
{json.dumps(own_output.to_dict(), ensure_ascii=False, indent=2)}

Найди их слабые места и улучши свой ответ."""
            
            try:
                raw = provider.generate(
                    system_prompt=rebuttal_system,
                    user_prompt=user_msg,
                    model_id=own_output.model_id,
                    temperature=0.6
                )
                
                parsed = safe_parse_json(raw)
                validate_solver_json(parsed)
                
                improved.append(SolverOutput(
                    model_id=own_output.model_id,
                    task_id=task_id,
                    final_answer=parsed["final_answer"],
                    confidence=float(parsed["confidence"]),
                    assumptions=list(parsed["assumptions"]),
                    risks=list(parsed["risks"]),
                    evidence=list(parsed["evidence"]),
                    self_checks=list(parsed["self_checks"]),
                    latency_ms=own_output.latency_ms
                ))
            except Exception as e:
                logger.error(f"Rebuttal for {own_output.model_id} failed: {e}")
                improved.append(own_output)
        
        return improved
