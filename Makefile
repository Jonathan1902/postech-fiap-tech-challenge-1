PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: setup install data lint format test test-cov run clean

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) scripts/build_datasets.py

lint:
	.venv/bin/ruff check src/ tests/ scripts/

format:
	.venv/bin/ruff format src/ tests/ scripts/

test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest --cov=src/churn_predictor --cov-report=term-missing tests/

run:
	$(PYTHON) -m uvicorn churn_predictor.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
