# MJEPA

MJEPA is a Python library implementing a JEPA-based architecture for self-supervised pre-training of vision
transformers.

## Requirements

- CPython 3.11 through 3.14.
- `uv>=0.11.28,<0.12`.
- CUDA is optional; CUDA-marked tests run only when a compatible device is available.

The package supports Python 3.11–3.14, while CircleCI validates Python 3.14.6.

## Installation

Set up a contributor environment with every development group and optional feature:

```bash
make init
```

For a runtime-only environment, use `uv sync`. Optional features can be installed independently:

```bash
uv sync --extra visualization
uv sync --extra dicom
```

The `visualization` extra provides Matplotlib-based analysis tools. The `dicom` extra provides TIFF/DICOM loading
through `dicom-preprocessing`.

## Configuration

[`config.yaml`](./config.yaml) contains a runnable CIFAR-10 ViT-S/4 example using the library's YAML constructors.
It defines the trainer, ViT backbone and heads, JEPA behavior, and optimizer settings.

## Development Commands

- `make check` formats the source, then runs lint, formatting, type, and test gates.
- `make style` applies Ruff lint fixes and formatting.
- `make quality` checks Ruff lint and formatting without rewriting source files.
- `make types` runs basedpyright over `mjepa/` and `tests/`.
- `make test` runs the full pytest suite with terminal coverage.
- `make test-ci` forces CPU execution, excludes `ci_skip` tests, and writes `coverage.xml`.
- `make test-<pattern>` runs tests matching a pytest expression, such as `make test-model`.
- `make test-pdb-<pattern>` runs a matching subset with PDB fallback.

CircleCI runs quality checks and the CPU test/type gates on Python 3.14.6. CUDA-marked tests should also be run
locally before changes that affect model execution or device/dtype behavior.
