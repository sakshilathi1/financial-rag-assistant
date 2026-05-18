.PHONY: install test test-all lint format run-api eval clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short --timeout=60

test-all:
	python -m pytest tests/ -v --tb=short --timeout=60 -m ""

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	python scripts/03_generate_eval_set.py
	python scripts/04_run_evaluation.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
