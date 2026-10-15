# KTTS

## Rules
- The sibling folder `../Kokoro-FastAPI/` is read-only: copy code from it, never edit it or import from it.
- When copying modules, record the source Kokoro-FastAPI commit in `README.md`.
- Use `uv` for everything: `uv run ...`, `uv add ...`. The venv is `.venv/`.
- Run tests with `uv run pytest`.
- Plan: `docs/New Project Plan.md`.
