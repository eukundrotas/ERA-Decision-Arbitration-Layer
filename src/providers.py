import requests
import json
import logging
import time
from typing import Dict, Any, Optional
from src.config import config

logger = logging.getLogger(__name__)

class OpenRouterProvider:
    def __init__(self):
        self.base_url = config.openrouter_base_url
        self.api_key = config.openrouter_api_key
        self.timeout = config.solver_timeout_sec
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Генерирует ответ от LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://era-dal.example.com",
            "X-Title": "ERA Decision & Arbitration Layer"
        }
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95
        }
        
        try:
            start = time.time()
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            latency_ms = int((time.time() - start) * 1000)
            
            logger.info(f"Model {model_id} responded in {latency_ms}ms")
            return content
            
        except requests.exceptions.Timeout:
            logger.error(f"Model {model_id} timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            raise

provider = OpenRouterProvider()
