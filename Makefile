.PHONY: help setup install test smoke train eval data clean lint

CONFIG ?= configs/train_small.yaml

help:
	@echo "Targets:"
	@echo "  make setup    - clean-environment bootstrap (install + test + smoke)"
	@echo "  make install  - pip install -r requirements.txt"
	@echo "  make test     - run unit tests"
	@echo "  make smoke    - tiny end-to-end run (configs/smoke.yaml)"
	@echo "  make data     - prepare/download datasets"
	@echo "  make train    - train (CONFIG=configs/train_small.yaml)"
	@echo "  make eval     - evaluate a checkpoint"
	@echo "  make clean    - remove caches and __pycache__"

setup:
	bash scripts/setup.sh

install:
	python -m pip install --upgrade pip && python -m pip install -r requirements.txt

test:
	python -m pytest -q

smoke:
	python scripts/train.py --config configs/smoke.yaml
	python scripts/evaluate.py --config configs/smoke.yaml

data:
	python scripts/prepare_data.py --config $(CONFIG)

train:
	python scripts/train.py --config $(CONFIG)

eval:
	python scripts/evaluate.py --config $(CONFIG)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
