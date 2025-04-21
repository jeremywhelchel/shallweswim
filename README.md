# shallweswim.today

Website to display swimming conditions at Coney Island / Brighton Beach

## Run locally (directly)

1. `poetry install`
1. `PORT=12345 poetry run python shallweswim/main.py`
1. Visit http://localhost:12345

## Run locally (via Docker)

1. `docker build -t shallweswim .`
1. `docker run -e PORT=80 -p 12345:80 shallweswim`
1. Visit http://localhost:12345

## Deploy

Hosted on Google Cloud Run

1. Run `./build_and_deploy.sh`

## Development

### Setup

```bash
# Install poetry from its website (`brew install` version seems problematic on mac)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up pre-commit hooks
poetry run pre-commit install
```

## Testing

The project uses pytest for both unit and integration tests:

### Unit Tests

Run unit tests (fast, no external dependencies):

```bash
# Run all unit tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_noaa.py
```

### Integration Tests

Integration tests connect to live external services (like the NOAA API) to verify compatibility. These tests are marked with `@pytest.mark.integration` and are skipped by default. They must be explicitly enabled with the `--run-integration` flag:

```bash
# Run all integration tests
poetry run pytest -v -m integration --run-integration

# Run both unit and integration tests
poetry run pytest -v --run-integration
```

Note: Integration tests may occasionally fail if external services are experiencing issues or if the expected data is temporarily unavailable.

## Code Quality

The project uses the following tools to maintain code quality, both locally and in CI:

### Code Formatting

```bash
# Check formatting without making changes
poetry run black --check .

# Format all Python files
poetry run black .
```

### Type Checking

```bash
# Run mypy type checking
poetry run mypy --config-file=pyproject.toml .
```

### Dead Code Detection

The project uses Pylint to detect unused imports, variables, and other forms of dead code. This check is included in the pre-commit hooks and will cause the commit to fail if unused code is detected.

```bash
# Check for unused imports and variables in a specific file
poetry run pylint --disable=all --enable=unused-import,unused-variable file.py

# Check the entire codebase for unused code
poetry run pylint --disable=all --enable=unused-import,unused-variable shallweswim/ tests/

# Run the pre-commit hook
poetry run pre-commit run pylint
```

The pylint configuration (`.pylintrc`) is specifically focused on detecting unused code while ignoring other stylistic issues that are already handled by Black and mypy.

### Other Tools

- **prettier**: Format HTML, Markdown, and YAML files
  ```bash
  npx prettier --write "**/*.{html,md,yaml,yml}"
  ```

## Continuous Integration

GitHub Actions workflows automatically verify the following on every push:

- **Unit Tests**: All unit tests pass
- **Type Checking**: No type errors are found by mypy
- **Code Formatting**: All Python code follows Black style
- **Dead Code Detection**: No unused imports or variables are present

Additionally, a separate integration test workflow runs daily to ensure compatibility with external APIs.
