# AGENTS.md

## Cursor Cloud specific instructions

### Overview
TumorTwin is a Python scientific library for image-guided cancer-patient digital twins. It is a single Python package (no microservices, no Docker, no database). All computation runs locally on CPU (GPU optional).

### Python version
The project requires **Python >=3.9, <3.12** (tested on 3.11). The Cloud VM ships with Python 3.12, which is **outside** the supported range. Python 3.11 is installed from the `deadsnakes` PPA and the virtualenv lives at `/workspace/.venv`.

### Activating the environment
```
source /workspace/.venv/bin/activate
```

### Common commands
See `Makefile` for canonical targets:
- `make lint` — runs `flake8` then `mypy` on `tumortwin/` and `tests/`
- `make format` — runs `black` then `isort`
- `make test` — runs `pytest tests`
- `make docs-serve` — serves MkDocs locally

### Known pre-existing issues
- **flake8 / mypy**: Both report pre-existing warnings/errors in the codebase. These are not regressions.
- **5 test files fail to collect** (`test_chemotherapy.py`, `test_diffusion.py`, `test_radiotherapy.py`, `test_reaction.py`, `test_qoi.py`) because they import modules (`tumortwin.solvers.predict_cell_count`, `tumortwin.qoi.qoi`) that no longer exist.
- **4 tests in `test_crop.py` fail** because they reference relative `data/` paths that are not present in the expected working directory.
- The remaining **15 tests pass** cleanly.

### Running a quick smoke test
```python
python -c "import tumortwin; print('OK')"
```
Or run the passing test subset:
```
pytest tests --ignore=tests/models/test_chemotherapy.py --ignore=tests/models/test_diffusion.py --ignore=tests/models/test_radiotherapy.py --ignore=tests/models/test_reaction.py --ignore=tests/test_qoi.py -v
```
