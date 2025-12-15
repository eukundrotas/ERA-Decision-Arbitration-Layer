# 📋 Инструкция по загрузке GitHub Actions Workflows

GitHub App интеграция не имеет прав на создание workflow файлов. 
Загрузите их вручную через GitHub Web UI.

## Шаг 1: Откройте репозиторий

https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer

## Шаг 2: Создайте директорию .github/workflows

1. Нажмите **Add file** → **Create new file**
2. В поле имени файла введите: `.github/workflows/ci.yml`
3. Скопируйте содержимое ниже

## Файл 1: ci.yml (Continuous Integration)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install black isort flake8
      - run: black --check --diff src/ tests/ app.py
        continue-on-error: true
      - run: isort --check-only --diff src/ tests/ app.py
        continue-on-error: true

  test:
    name: Tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - run: python -m pytest tests/ -v --cov=src
        env:
          OPENROUTER_API_KEY: "test-key"

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          target: production
          push: false
          tags: era-dal:test
```

## Файл 2: release.yml (Auto Release)

Создайте второй файл: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release'
        required: true

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          pip install -r requirements.txt pytest
          python -m pytest tests/ -v
        env:
          OPENROUTER_API_KEY: "test-key"

  docker:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-qemu-action@v3
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          target: production
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}

  release:
    needs: [test, docker]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - run: |
          tar -czf era-dal.tar.gz --exclude='.git' .
          zip -r era-dal.zip . -x "*.git*"
      - uses: softprops/action-gh-release@v1
        with:
          files: |
            era-dal.tar.gz
            era-dal.zip
          generate_release_notes: true
```

## Шаг 3: Commit

После создания каждого файла нажмите **Commit new file**.

## Проверка

После загрузки перейдите в **Actions** tab репозитория:
https://github.com/eukundrotas/ERA-Decision-Arbitration-Layer/actions

CI workflow автоматически запустится при следующем push в main.
