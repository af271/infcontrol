# infcontrol

Symbolic inference-control utilities and feature operators.

## Local development

1. Create and activate a virtual environment.
2. Install William from the sibling repository in editable mode.
3. Install this repository in editable mode with dev dependencies.
4. Run tests.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../../william
pip install -e .[dev]
pytest
```
