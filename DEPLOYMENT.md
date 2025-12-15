# ERA DAL - Deployment Guide

## 📦 Получение проекта

Проект создан в sandbox по пути: `/home/user/webapp/era-dal/`

Доступны архивы:
- **tar.gz**: `/home/user/webapp/era-dal-v1.0.0.tar.gz` (92 KB)
- **zip**: `/home/user/webapp/era-dal-v1.0.0.zip` (135 KB)

## 🚀 Быстрый старт

### 1. Распаковка

```bash
# Вариант 1: tar.gz
tar -xzf era-dal-v1.0.0.tar.gz
cd era-dal

# Вариант 2: zip
unzip era-dal-v1.0.0.zip
cd era-dal
```

### 2. Настройка окружения

```bash
# Создать виртуальное окружение (рекомендуется)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### 3. Конфигурация

```bash
# Скопировать шаблон
cp .env.example .env

# Отредактировать .env и добавить API ключ
nano .env  # или любой редактор
```

Обязательно установить:
```env
OPENROUTER_API_KEY=sk-or-v1-...
```

### 4. Проверка установки

```bash
# Запустить тесты
python -m pytest tests/ -v

# Проверить CLI
python app.py --help
```

## 📝 Примеры использования

### Одна задача

```bash
python app.py \
  --pool science \
  --problem "Объясни, как работает фотосинтез" \
  --repeats 5 \
  --consensus-topk 3 \
  --epsilon 0.07 \
  --rebuttal
```

### Файл с задачами

```bash
python app.py \
  --pool math \
  --problems-file examples/sample_problems.txt \
  --repeats 3 \
  --out-dir ./results
```

### Только hard select (без консенсуса)

```bash
python app.py \
  --pool econ \
  --problem "Как инфляция влияет на экономику?" \
  --hard-only \
  --repeats 5
```

## 📊 Результаты

После выполнения проверьте директорию `out/`:

- **runs.csv** — детальные логи каждого solver
- **runs.xlsx** — Excel версия (требует openpyxl)
- **final.json** — финальное решение + stability metrics
- **model_quality.json** — память качества моделей

Пример `final.json`:
```json
{
  "task_id": "task_001",
  "final_answer": "...",
  "decision_mode": "consensus_top3",
  "used_candidates": ["openai/gpt-4-turbo-preview", "anthropic/claude-3-opus"],
  "stability": {
    "majority_rate": 0.8,
    "ci_lower": 0.6,
    "ci_upper": 0.95,
    "total_runs": 5
  }
}
```

## 🔧 Конфигурация пулов

Система поддерживает 4 домена:

| Домен | Описание | Кол-во моделей |
|-------|----------|----------------|
| **science** | Научные вопросы, анализ | 7 |
| **math** | Математика, логика | 6 |
| **med** | Медицина, биология | 5 |
| **econ** | Экономика, финансы | 6 |

Модели можно настроить в `src/solver_pool.py`.

## 🐛 Устранение проблем

### API ключ не найден
```
ValueError: OPENROUTER_API_KEY не установлен
```
**Решение**: Проверьте `.env` файл или установите через:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Timeout ошибки
```
requests.exceptions.Timeout
```
**Решение**: Увеличьте timeout в `.env`:
```env
SOLVER_TIMEOUT_SEC=120
ARBITER_TIMEOUT_SEC=60
```

### Зависимости не установлены
```
ModuleNotFoundError: No module named 'dotenv'
```
**Решение**: 
```bash
pip install -r requirements.txt
```

## 📚 Дополнительная информация

- **README.md** — общая документация
- **whitepaper.md** — техническая документация
- **tests/** — unit тесты (26 тестов)
- **examples/** — примеры задач

## 🔗 GitHub

Для загрузки на GitHub:

```bash
cd era-dal

# Инициализация (уже сделано)
git init
git add .
git commit -m "Initial commit: ERA DAL v1.0.0"

# Создать репозиторий на GitHub
gh repo create era-dal --public --description "ERA Decision & Arbitration Layer"

# Или вручную на github.com, затем:
git remote add origin https://github.com/YOUR_USERNAME/era-dal.git
git branch -M main
git push -u origin main
```

## 📞 Поддержка

- **Документация**: см. README.md и whitepaper.md
- **Тесты**: `python -m pytest tests/ -v`
- **Версия**: 1.0.0
- **Дата**: 2025-12-15

---

**Готово к production использованию!** 🚀
