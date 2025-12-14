import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelMemory:
    """ERA-style eval layer — память качества моделей"""
    
    def __init__(self, memory_path: str = None):
        self.memory_path = memory_path or "./out/model_quality.json"
        self.state = self._load()
    
    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Загружает память с диска"""
        path = Path(self.memory_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model_quality.json: {e}")
        return {}
    
    def save(self):
        """Сохраняет память на диск"""
        Path(self.memory_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def update_reliability(self, model_id: str, reward: float, lr: float = 0.05):
        """
        Обновляет reliability модели (EMA).
        reward: 1.0 если модель в финале, 0.0 иначе
        """
        if model_id not in self.state:
            self.state[model_id] = {"n": 0, "reliability": 0.5}
        
        s = self.state[model_id]
        s["n"] += 1
        s["reliability"] = (1 - lr) * float(s["reliability"]) + lr * float(reward)
        
        logger.info(f"Updated {model_id}: n={s['n']}, reliability={s['reliability']:.3f}")
    
    def get_reliability(self, model_id: str) -> float:
        """Получает reliability модели"""
        if model_id in self.state:
            return float(self.state[model_id].get("reliability", 0.5))
        return 0.5
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает всё состояние"""
        return self.state.copy()
