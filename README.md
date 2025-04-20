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

Setup with:

```
# Install poetry from its website (`brew install` version seems problematic on mac)
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

Tools used:

- Format python code with `black`
- Type check with `mypy`
  `poetry run mypy --config-file=pyproject.toml .`
- HTML/MD/Yaml formatted with `prettier`
