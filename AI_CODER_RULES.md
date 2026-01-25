# AI CODER RULES

**For full documentation, see [README.md](README.md).** This file contains quick-reference rules for AI coders.

## Project Overview

Open water swimming conditions app (shallweswim.today). Aggregates tide, current, and temperature data from NOAA/USGS APIs for multiple locations.

## Essential Commands

```bash
uv run pytest -v -k "not integration"   # Unit tests (default)
uv run mypy --config-file=pyproject.toml .  # Type checking
uv run pre-commit run --all-files       # All quality checks
uv run python -m shallweswim.main --port=12345  # Run locally
```

## Critical Rules

1. **Use uv for everything** - Never use pip, poetry, or bare python. Always `uv run <command>`.
2. **Run commands from repo root** - Don't cd into subdirectories.
3. **Run pre-commit after major changes** - Catches formatting, linting, and type errors.
4. **Don't run integration tests** unless explicitly requested - They hit live external APIs.
5. **Follow CONVENTIONS.md strictly** - See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for architecture and coding standards.

## Architecture Quick Reference

```
API Handler → LocationDataManager → Feed → ApiClient → External Service
```

- **api.py**: Route handlers only, delegates to data.py
- **data.py**: LocationDataManager coordinates feeds per location
- **feeds.py**: Data fetching with caching/expiration
- **clients/**: Pure HTTP clients, no business logic
- **config.py**: All location configs and station IDs

## Before Implementing

- Check existing patterns in similar code before writing new code
- New locations: config.py → data.py → api.py → main.py → tests
- CPU-bound work (plotting): Must use process pool, not async
