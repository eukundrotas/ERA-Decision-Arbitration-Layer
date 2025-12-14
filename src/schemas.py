from jsonschema import validate, ValidationError
from typing import Any, Dict

SOLVER_SCHEMA = {
    "type": "object",
    "required": [
        "model_id",
        "task_id",
        "final_answer",
        "confidence",
        "assumptions",
        "risks",
        "evidence",
        "self_checks"
    ],
    "additionalProperties": False,
    "properties": {
        "model_id": {"type": "string"},
        "task_id": {"type": "string"},
        "final_answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "self_checks": {"type": "array", "items": {"type": "string"}}
    }
}

ARBITER_SCHEMA = {
    "type": "object",
    "required": ["task_id", "selected_model_id", "ranking", "final_answer", "arbiter_notes"],
    "additionalProperties": True,
    "properties": {
        "task_id": {"type": "string"},
        "selected_model_id": {"type": "string"},
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["model_id", "score"],
                "additionalProperties": False,
                "properties": {
                    "model_id": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        },
        "final_answer": {"type": "string"},
        "arbiter_notes": {"type": "array", "items": {"type": "string"}}
    }
}

def validate_solver_json(obj: Dict[str, Any]) -> None:
    try:
        validate(instance=obj, schema=SOLVER_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Solver JSON validation error: {e.message}")

def validate_arbiter_json(obj: Dict[str, Any]) -> None:
    try:
        validate(instance=obj, schema=ARBITER_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Arbiter JSON validation error: {e.message}")
