# several pytest settings
WORLD_SIZE ?= 1  # world size for launcher tests
PYTHON ?= python3  # Python command
PYTEST ?= pytest  # Pytest command
PYRIGHT ?= pyright  # Pyright command. Pyright must be installed seperately -- e.g. `node install -g pyright`
EXTRA_ARGS ?=  # extra arguments for pytest
EXTRA_LAUNCHER_ARGS ?= # extra arguments for the composer cli launcher

test:
	$(PYTHON) -m $(PYTEST) $(EXTRA_ARGS)

test-gpu:
	$(PYTHON) -m $(PYTEST) -m gpu $(EXTRA_ARGS)

.PHONY: test test-gpu test-dist test-dist-gpu
