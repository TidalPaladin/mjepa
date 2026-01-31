.PHONY: clean clean-env check quality style tag-version test env upload upload-test

PROJECT=mjepa
QUALITY_DIRS=$(PROJECT) tests
CLEAN_DIRS=$(PROJECT) tests
PYTHON=uv run python

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

clean: ## remove cache files
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-env: ## remove the virtual environment directory
	rm -rf .venv

init: ## pulls submodules and initializes virtual environment
	git submodule update --init --recursive
	which uv || pip install --user uv
	uv sync --all-groups

quality:
	$(MAKE) clean
	$(PYTHON) -m ruff check $(QUALITY_DIRS)
	$(PYTHON) -m ruff format --check $(QUALITY_DIRS)

style:
	$(PYTHON) -m ruff check --fix $(QUALITY_DIRS)
	$(PYTHON) -m ruff format $(QUALITY_DIRS)

test: ## run unit tests
	$(PYTHON) -m pytest \
		-rs \
		--cov=./$(PROJECT) \
		--cov-report=term \
		./tests/

test-%: ## run unit tests matching a pattern
	$(PYTHON) -m pytest -s -r fE -k $* ./tests/ --tb=no

test-pdb-%: ## run unit tests matching a pattern with PDB fallback
	$(PYTHON) -m pytest -rs --pdb -k $* -v ./tests/ 

test-ci: ## runs CI-only tests
	export "CUDA_VISIBLE_DEVICES=''" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "not ci_skip" \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		./tests/

types: ## run basedpyright type checking
	$(PYTHON) -m basedpyright $(QUALITY_DIRS)

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
