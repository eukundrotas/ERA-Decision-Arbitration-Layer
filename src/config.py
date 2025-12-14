import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Defaults
    default_pool: str = os.getenv("DEFAULT_POOL", "science")
    default_repeats: int = int(os.getenv("DEFAULT_REPEATS", "5"))
    default_consensus_topk: int = int(os.getenv("DEFAULT_CONSENSUS_TOPK", "3"))
    default_epsilon: float = float(os.getenv("DEFAULT_EPSILON", "0.07"))
    default_rebuttal: bool = os.getenv("DEFAULT_REBUTTAL", "true").lower() == "true"
    default_out_dir: str = os.getenv("DEFAULT_OUT_DIR", "./out")
    
    # Timeouts
    solver_timeout_sec: int = int(os.getenv("SOLVER_TIMEOUT_SEC", "60"))
    arbiter_timeout_sec: int = int(os.getenv("ARBITER_TIMEOUT_SEC", "30"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def validate(self):
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY не установлен")
        Path(self.default_out_dir).mkdir(parents=True, exist_ok=True)

config = Config()
# Отложим валидацию до момента использования
# config.validate()
