import json
import logging
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

def safe_parse_json(raw_str: str) -> Dict[str, Any]:
    """Безопасный парсинг JSON с попыткой очистки"""
    raw_str = raw_str.strip()
    
    # Если есть markdown-обёртка, удалим
    if raw_str.startswith("```json"):
        raw_str = raw_str[7:]
    if raw_str.startswith("```"):
        raw_str = raw_str[3:]
    if raw_str.endswith("```"):
        raw_str = raw_str[:-3]
    
    raw_str = raw_str.strip()
    
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError(f"Failed to parse JSON: {e}")

def setup_logging(log_level: str = "INFO"):
    """Инициализирует логирование"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def ensure_output_dir(out_dir: str):
    """Создаёт директорию для артефактов"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def write_run_record(record, out_dir: str):
    """Добавляет строку в runs.csv"""
    csv_path = Path(out_dir) / "runs.csv"
    
    should_write_header = not csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pool', 'iter_index', 'round_number', 'solver_model_id',
            'latency_ms', 'confidence', 'arbiter_score', 'decision_mode',
            'used_candidates', 'final_answer', 'notes', 'timestamp'
        ])
        if should_write_header:
            writer.writeheader()
        writer.writerow(record.__dict__)

def write_final_json(final_data: Dict[str, Any], out_dir: str):
    """Записывает final.json"""
    json_path = Path(out_dir) / "final.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

def write_model_quality(quality_data: Dict[str, Any], out_dir: str):
    """Записывает model_quality.json"""
    json_path = Path(out_dir) / "model_quality.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(quality_data, f, ensure_ascii=False, indent=2)

def csv_to_xlsx(out_dir: str):
    """Конвертирует runs.csv в runs.xlsx"""
    try:
        csv_path = Path(out_dir) / "runs.csv"
        xlsx_path = Path(out_dir) / "runs.xlsx"
        
        df = pd.read_csv(csv_path)
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        logger.info(f"Wrote {xlsx_path}")
    except Exception as e:
        logger.warning(f"Could not create XLSX: {e}")
