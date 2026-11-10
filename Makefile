SRCS_DIR = src
UV_PY = uv run

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	@export HF_HOME=/sgoinfre/mbichet/hf
	uv sync --cache-dir ~/goinfre/.uv_cache 

install-dataset:
	@mkdir -p data/raw
	@unzip vllm-0.10.1.zip -d data/raw
	@echo "vllm-0.10.1.zip extracted to data/raw/vllm-0.10.1"

run:
	@$(UV_PY) -m src $(ARGS)

debug:
	@$(UV_PY) -m src -v $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf data/processed
	rm -rf data/output

fclean: clean
	rm -rf .venv

lint:
	$(UV_PY) flake8 $(SRCS_DIR)
	$(UV_PY) mypy $(SRCS_DIR) $(MYPY_FLAGS)

.PHONY: install run debug clean fclean lint lint-strict install-dataset