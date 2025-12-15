# ERA Decision & Arbitration Layer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-26%20passed-success.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Надёжное принятие решений через ансамбль LLM, арбитраж, консенсус и самокритику.**

🔗 **GitHub**: [https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer](https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer)

## Что это?

ERA DAL — модуль принятия решений для научных, аналитических и экспертных задач,
где критичны:

- **Устойчивость** — стабильные результаты при повторных запусках
- **Контроль галлюцинаций** — явная фиксация рисков и допущений
- **Воспроизводимость** — полный audit trail
- **Доверие** — количественная оценка надёжности (Wilson CI)

## Архитектура

```
Problem
  ↓
Solver Pool (5–12 моделей, параллельно)
  ↓
Disagreement Detector
  ↓
Arbiter Ranker (оценка качества)
  ├─ Hard Select (одно решение)
  └─ Consensus Synthesizer (top-2/top-3)
  ↑
  Rebuttal Round (если disagreement высокий)
  ↓
Multi-Run Stability (Wilson CI)
  ↓
Final Answer + Artifacts
```

## Быстрый старт

### 1. Установка

```bash
git clone https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer.git
cd ERA-Decision-Arbitration-Layer

cp .env.example .env
# Заполните OPENROUTER_API_KEY в .env

pip install -r requirements.txt
```

### 2. Запуск

```bash
export OPENROUTER_API_KEY="your_api_key"

# Одна задача
python app.py --pool science --problem "Объясни, как работает фотосинтез"

# Файл с задачами
python app.py --pool science --problems-file examples/sample_problems.txt --repeats 5 --consensus-topk 3 --rebuttal

# Только hard select (без консенсуса и rebuttal)
python app.py --pool science --problem "..." --hard-only
```

### 3. Результаты

Артефакты в `./out/`:

- **runs.csv** — детальные логи каждого solver
- **runs.xlsx** — Excel-версия (опционально)
- **final.json** — финальные решения + stability metrics
- **model_quality.json** — память качества моделей (для обучения системы)

## CLI Аргументы

| Аргумент | Описание | По умолчанию |
|----------|---------|-------------|
| `--pool` | science / math / med / econ | science |
| `--repeats` | Число прогонов для stability | 5 |
| `--consensus-topk` | 2 или 3 | 3 |
| `--epsilon` | Порог для консенсуса (gap между top1 и top2) | 0.07 |
| `--rebuttal` | Включить rebuttal | True |
| `--hard-only` | Только hard select | False |
| `--problem` | Одна задача строкой | - |
| `--problems-file` | Файл с задачами | - |
| `--out-dir` | Каталог артефактов | ./out |

## Режимы принятия решений

### hard_select
Арбитр выбирает одно лучшее решение по критериям качества.

### consensus_top2 / consensus_top3
Если gap между top1 и top2 < epsilon, система синтезирует финальный ответ
из top-K кандидатов.

### rebuttal
Если disagreement_rate ≥ threshold, каждый solver получает ответы других,
критикует их и улучшает свой ответ. Затем повторный арбитраж.

## Домены (пулы моделей)

- **science** — сбалансированный пул (логика + объяснимость + критика)
- **math** — усиление формальной логики, низкая температура
- **med** — высокая осторожность, редакторский стиль
- **econ** — структура, аккуратность, ясность

## Стабильность и доверие

Система прогоняет задачу N раз и считает:

- **majority_rate** — как часто одно и то же решение
- **95% Wilson CI** — доверительный интервал
- **mode_distribution** — распределение decision_mode

Пример `final.json`:

```json
{
  "task_id": "task_001",
  "final_answer": "...",
  "decision_mode": "consensus_top3",
  "stability": {
    "majority_rate": 0.8,
    "ci_lower": 0.6,
    "ci_upper": 0.95,
    "total_runs": 5
  }
}
```

## Память качества моделей

После каждого запуска система обновляет `model_quality.json`:

```json
{
  "openai/gpt-4-turbo-preview": {
    "n": 120,
    "reliability": 0.72
  },
  "anthropic/claude-3-opus": {
    "n": 120,
    "reliability": 0.68
  }
}
```

Reliability используется для взвешенного консенсуса и адаптации под домен.

## Структура проекта

```
era-dal/
├── .env.example         # Пример конфигурации
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
├── whitepaper.md        # Подробная документация
│
├── src/
│   ├── __init__.py
│   ├── config.py        # Конфигурация + env
│   ├── schemas.py       # JSON schemas + валидация
│   ├── prompts.py       # Промпты по доменам
│   ├── providers.py     # OpenRouter provider
│   ├── models.py        # Dataclasses
│   ├── solver_pool.py   # Ансамбль solvers
│   ├── arbiter.py       # Модель-арбитр
│   ├── consensus.py     # Синтезатор консенсуса
│   ├── rebuttal.py      # Rebuttal-раунд
│   ├── stability.py     # Multi-run + Wilson CI
│   ├── model_memory.py  # ERA-style eval layer
│   ├── orchestrator.py  # Главный оркестратор
│   └── utils.py         # Helpers
│
├── app.py               # CLI entry point
├── tests/               # Тесты
├── examples/            # Примеры задач
└── out/                 # Артефакты
```

## Развитие

Без смены концепции можно добавить:

- [ ] Асинхронное масштабирование (asyncio)
- [ ] Формальные метрики корректности (Levenshtein, BLEU, ROUGE)
- [ ] Доменные бенчи для калибровки
- [ ] Экспорт в dashboards (Grafana, Tableau)
- [ ] REST API wrapper
- [ ] Web UI

## Лицензия

MIT

---

**Автор:** Eugene Kundrotas  
**Версия:** 1.0.0  
**Дата:** December 15, 2025

📧 **Contact**: [GitHub Issues](https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer/issues)
