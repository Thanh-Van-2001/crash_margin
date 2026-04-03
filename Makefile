.PHONY: install test lint figures experiments clean all

PYTHON ?= python

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check crashmargin/ experiments/ scripts/ tests/

# ── Experiments (Tables 1-3 + Figure 5) ──────────────────────────────────
experiments: classification ablation margin walkforward

classification:
	$(PYTHON) experiments/run_classification.py --seed 42

ablation:
	$(PYTHON) experiments/run_ablation.py --seed 42

margin:
	$(PYTHON) experiments/run_margin_eval.py --seed 42

walkforward:
	$(PYTHON) experiments/run_walkforward.py --seed 42

# ── Figures ───────────────────────────────────────────────────────────────
figures:
	$(PYTHON) scripts/generate_figures.py --output_dir outputs/figures

# ── Full pipeline ─────────────────────────────────────────────────────────
all: install experiments figures test

clean:
	rm -rf outputs/ build/ dist/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
