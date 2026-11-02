VENV        = .venv

SRC_DIR     = src
MAIN        = $(SRC_DIR)/main.py

FLAKE = flake8
MYPY  = mypy

MYPY_FLAGS = \
	--warn-return-any           \
	--warn-unused-ignores       \
	--ignore-missing-imports    \
	--disallow-untyped-defs     \
	--check-untyped-defs

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size 2000
	uv run python -m src answer "How to configure the OpenAI server?" --k 5

lint: install
	uv run $(FLAKE) $(SRC_DIR) --exclude $(VENV)
	uv run $(MYPY) $(SRC_DIR) $(MYPY_FLAGS)

lint-strict: install
	uv run $(FLAKE) $(SRC_DIR) --exclude $(VENV)
	uv run $(MYPY) $(SRC_DIR) --strict

debug: install
	uv run -m pdb $(MAIN)


clean:
	uv clean
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-package:
	rm -rf build dist

.PHONY: install run debug package test lint lint-strict profiler clean clean-package