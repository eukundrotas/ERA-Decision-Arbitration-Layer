# ERA Decision & Arbitration Layer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-40%2B%20passed-success.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)](.github/workflows/)

**Надёжное принятие решений через ансамбль LLM, арбитраж, консенсус и самокритику.**

🔗 **GitHub**: [https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer](https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer)

## ✨ Что нового в v1.1.0

- 🧠 **Semantic Clustering** — кластеризация ответов по семантике для точного определения разногласий
- ⏱️ **Adaptive Early Stopping** — оптимизация количества запусков на основе конвергенции (экономия API)
- 📊 **Real-time Dashboard** — веб-дашборд для мониторинга в реальном времени
- 🐳 **Docker** — готовые образы для production и development
- ⚙️ **CI/CD** — GitHub Actions для автоматического тестирования и релизов

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
🆕 Semantic Clustering (кластеризация ответов)
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
🆕 Early Stopping Check (если конвергенция достигнута)
  ↓
Final Answer + Artifacts
  ↓
🆕 Dashboard Metrics (real-time)
```

## 🚀 Быстрый старт

### Вариант 1: Локальная установка

```bash
git clone https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer.git
cd ERA-Decision-Arbitration-Layer

cp .env.example .env
# Заполните OPENROUTER_API_KEY в .env

pip install -r requirements.txt

# Запуск
python app.py --pool science --problem "Объясни, как работает фотосинтез"
```

### Вариант 2: Docker 🐳

```bash
# Клонировать репозиторий
git clone https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer.git
cd ERA-Decision-Arbitration-Layer

# Создать .env файл
cp .env.example .env
# Добавить OPENROUTER_API_KEY в .env

# Запустить с Docker Compose
docker-compose run cli --pool science --problem "Объясни, как работает фотосинтез"

# Или запустить Dashboard
docker-compose up dashboard
# Открыть http://localhost:8080
```

### Вариант 3: Docker напрямую

```bash
# Собрать образ
docker build -t era-dal:latest .

# Запустить CLI
docker run -e OPENROUTER_API_KEY=sk-your-key era-dal:latest \
  --pool science --problem "Что такое ДНК?"

# Запустить Dashboard
docker run -p 8080:8080 -e OPENROUTER_API_KEY=sk-your-key \
  --entrypoint python era-dal:latest -m src.api 8080
```

## 📊 Dashboard

Запустите Dashboard для мониторинга в реальном времени:

```bash
# Локально
python -m src.api 8080

# Или через Docker
docker-compose up dashboard
```

Откройте http://localhost:8080 для просмотра:
- 📈 Статистика по проблемам и запускам
- 🤖 Производительность моделей (latency, confidence)
- 📝 Последние события в реальном времени
- ✅ Healthcheck API

### Dashboard API Endpoints

| Endpoint | Описание |
|----------|----------|
| `GET /api/health` | Health check |
| `GET /api/dashboard` | Полная статистика |
| `GET /api/events?limit=50` | Последние события |
| `GET /api/models` | Статистика по моделям |
| `GET /api/session/{id}` | Детали сессии |

## 🧠 Level 1 Upgrades

### Semantic Clustering (`src/embeddings.py`)

Кластеризует ответы по семантическому сходству для определения реальных разногласий
(не поверхностных различий в формулировках).

```python
from src.embeddings import analyze_disagreement

result = analyze_disagreement(
    answers=["The sky is blue due to Rayleigh scattering", 
             "Blue color comes from light scattering",
             "The ocean reflects the sky color"],
    model_ids=["gpt-4", "claude", "llama"],
    threshold=0.6
)

print(f"Clusters: {result.num_clusters}")
print(f"Disagreement: {result.disagreement_score:.2f}")
print(f"Recommendation: {result.recommendation}")  # 'hard_select', 'consensus', or 'rebuttal'
```

### Adaptive Early Stopping (`src/early_stopping.py`)

Оптимизирует количество запусков, останавливаясь раньше при достижении конвергенции.

```python
from src.early_stopping import check_early_stop

for run in range(1, max_runs + 1):
    answer = run_solver(problem)
    
    decision = check_early_stop(answer, run, max_runs)
    
    if decision.should_stop:
        print(f"Early stop at run {run}: {decision.reason}")
        print(f"Saved {decision.saved_runs} API calls!")
        break
```

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

## Результаты

Артефакты в `./out/`:

- **runs.csv** — детальные логи каждого solver
- **runs.xlsx** — Excel-версия (опционально)
- **final.json** — финальные решения + stability metrics
- **model_quality.json** — память качества моделей (для обучения системы)

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

## 🐳 Docker Services

| Service | Описание | Порт |
|---------|----------|------|
| `cli` | Основной CLI для запуска задач | - |
| `dashboard` | Real-time мониторинг | 8080 |
| `test` | Запуск тестов | - |
| `batch` | Пакетная обработка задач | - |

```bash
# Запустить тесты в Docker
docker-compose run test

# Запустить batch обработку
docker-compose run batch
```

## ⚙️ CI/CD

GitHub Actions автоматически:

1. **Lint & Format** — проверка Black, isort, flake8
2. **Unit Tests** — Python 3.9, 3.10, 3.11, 3.12
3. **Integration Tests** — с моками API
4. **Docker Build** — проверка сборки образа
5. **Security Scan** — Bandit, Safety

При создании тега `v*.*.*`:
- Создаётся GitHub Release
- Публикуются Docker образы в GHCR
- Генерируются архивы .tar.gz и .zip

## Структура проекта

```
era-dal/
├── .github/workflows/   # 🆕 CI/CD pipelines
│   ├── ci.yml          # Continuous Integration
│   └── release.yml     # Release automation
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
│   ├── utils.py         # Helpers
│   ├── embeddings.py    # 🆕 Semantic clustering
│   ├── early_stopping.py# 🆕 Adaptive early stopping
│   └── api.py           # 🆕 Dashboard API
│
├── tests/               # Unit tests (40+ tests)
├── examples/            # Примеры задач
├── out/                 # Артефакты
│
├── app.py               # CLI entry point
├── Dockerfile           # 🆕 Multi-stage Docker build
├── docker-compose.yml   # 🆕 Docker Compose config
├── requirements.txt
├── setup.py
├── README.md
└── whitepaper.md        # Подробная документация
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# С coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Конкретный модуль
python -m pytest tests/test_embeddings.py -v
python -m pytest tests/test_early_stopping.py -v
python -m pytest tests/test_api.py -v
```

## Домены (пулы моделей)

- **science** — сбалансированный пул (логика + объяснимость + критика)
- **math** — усиление формальной логики, низкая температура
- **med** — высокая осторожность, редакторский стиль
- **econ** — структура, аккуратность, ясность

## Roadmap

### ✅ Level 1 (v1.1.0)
- [x] Semantic Similarity Clustering
- [x] Adaptive Early Stopping
- [x] Real-time Dashboard API
- [x] Docker containerization
- [x] CI/CD with GitHub Actions

### 📋 Level 2 (Planned)
- [ ] Multi-Metric Evaluation (BLEU, ROUGE, BERTScore)
- [ ] A/B Testing framework
- [ ] Response caching
- [ ] Distributed execution
- [ ] Human feedback integration

### 🔮 Level 3 (Future)
- [ ] Prompt optimization
- [ ] Multi-language support
- [ ] Explainability module
- [ ] Domain benchmarks
- [ ] Quality assurance checks

## Лицензия

MIT

---

**Автор:** Eugene Kundrotas  
**Версия:** 1.1.0  
**Дата:** December 15, 2025

📧 **Contact**: [GitHub Issues](https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer/issues)
