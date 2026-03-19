# Repository Guidelines

## Project Structure & Module Organization
The repository is currently a small Python 3.12 project.

- `main.py`: current entry point.
- `pyproject.toml`: project metadata and Python requirement.
- `models/`: local model assets used by tooling. Keep large model files here and document additions in `models/README.md`.
- `README.md`: top-level project overview. Expand this when user-facing behavior is added.

For new application code, prefer creating a dedicated package directory instead of growing `main.py`. For example: `data_process/ingest.py` or `data_process/pipeline/`.

## Build, Test, and Development Commands
- If `uv` is not installed, install it first:
  - macOS: `brew install uv`
  - Cross-platform installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Verify installation: `uv --version`
- `uv run python main.py`: run the current entry point with the project environment.
- `uv sync`: create or update the local environment from `pyproject.toml`.
- `uv add <package>`: add a runtime dependency and update the lockfile.
- `uv run python -m compileall .`: quick syntax check before committing.

Use `uv` for dependency management and command execution in this repository. Avoid direct `pip` usage unless there is a specific maintenance reason.

## Coding Style & Naming Conventions
Follow standard Python conventions:

- Use 4-space indentation.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes.
- Keep functions small and focused; move reusable logic out of `main.py`.
- Prefer type hints for new functions and public interfaces.

Keep filenames descriptive, for example `text_cleaner.py` or `model_loader.py`.

## Testing Guidelines
There is no committed test suite yet. Add tests under `tests/` as features are introduced.

- Name test files `test_<module>.py`.
- Name test functions `test_<behavior>()`.
- Prefer `pytest` for new tests.
- Run tests with `uv run pytest` once `pytest` is added to project dependencies.

Include at least one happy-path test and one failure-path test for new logic.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Initial commit` and `Reset project`. Continue with concise, single-purpose messages, for example `Add model loading utility`.

For pull requests:

- Summarize the change and its purpose.
- Link the relevant issue or task when available.
- List validation steps you ran locally.
- Include screenshots only when UI or generated artifacts are affected.

## Security & Configuration Tips
Do not commit `.env`, virtual environments, caches, or model binaries outside `models/`. Keep secrets in local environment variables and document required configuration in `README.md`.
