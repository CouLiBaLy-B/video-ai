.PHONY: install install-web dev-api dev-web doctor test lint typecheck build-web quality clean

install:
	python -m pip install -e '.[dev]'

install-web:
	cd apps/web && npm install

dev-api:
	PYTHONPATH=apps/api/src uvicorn video_ai.main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	cd apps/web && npm run dev

doctor:
	python scripts/doctor.py

test:
	python -m pytest apps/api/tests

lint:
	python -m ruff check apps/api/src apps/api/tests migrations

typecheck:
	python -m mypy apps/api/src/video_ai

build-web:
	cd apps/web && npm run build

quality: doctor lint typecheck test build-web

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache apps/web/dist apps/web/tsconfig.tsbuildinfo
	find apps -type d -name __pycache__ -prune -exec rm -rf {} +
