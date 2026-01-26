# Shall We Swim Today?

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-Managed-blueviolet)](https://github.com/astral-sh/uv)

**A web application that helps open water swimmers make informed decisions about swim conditions.**

[shallweswim.today](https://shallweswim.today) aggregates real-time tide, current, and temperature data from government APIs (NOAA, USGS) for popular open water swimming locations:

- **New York** - Coney Island / Brighton Beach
- **San Diego** - La Jolla Cove
- **Chicago** - Ohio Street Beach
- **San Francisco** - Aquatic Park
- **Louisville** - Community Boathouse (Ohio River)
- **Austin** - Barton Springs
- **Boston** - L Street Beach
- **Seattle** - Alki Beach

## Features

- **Real-time conditions** from NOAA CO-OPS, NOAA NDBC, and USGS NWIS APIs
- **Tide predictions** with high/low tide times and heights
- **Current velocity** data with flood/ebb direction
- **Water temperature trends** (48-hour, 2-month, and multi-year)
- **Transit information** for NYC locations (subway status and alerts)
- **Mobile-friendly interface** for on-the-go swimmers
- **JSON API** for programmatic access to swim conditions

## Architecture

Shall We Swim is a FastAPI application with a modular architecture.

### Runtime Model

The application is **fully stateless** with no database or persistent storage. On startup, each instance fetches historical data (~8 days) directly from external APIs and holds it in memory. This design was chosen for Cloud Run deployment simplicity - instances can scale to zero and spin back up without managing storage.

**Implications:**

- Cold starts require fetching all data before serving (gated by `/api/healthy`)
- Instance shutdown loses all data (automatically re-fetched on next startup)
- Multiple instances don't share state (each fetches independently)

### Request Flow

**User requests** always serve cached data (fast, no external calls):
```
HTTP Request → API Handler → LocationDataManager → Feed (cached) → Response
```

**Background tasks** refresh feeds on intervals (10 min to 24 hours):
```
Background Task → Feed → ApiClient → External API → Update Cache
```

Failed fetches leave data stale until the next successful refresh.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation, coding standards, and error handling patterns.

## Getting Started

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker (optional, for containerized deployment)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/jeremywhelchel/shallweswim.git
cd shallweswim

# Install dependencies
uv sync

# Run the development server
uv run python -m shallweswim.main --port=12345
```

Then visit http://localhost:12345 in your browser.

### Run with Docker

```bash
# Build the Docker image
docker buildx build -t shallweswim .

# Run the container
docker run -e PORT=80 -p 12345:80 shallweswim
```

Then visit http://localhost:12345 in your browser.

## Deployment

The application is hosted on Google Cloud Run:

```bash
# Deploy to Google Cloud Run
./build_and_deploy.sh
```

## Development

**Documentation**

- [Architecture & Conventions](ARCHITECTURE.md) - **Read this first** before making changes.

### Setup

```bash
# Install uv (recommended method)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (including dev dependencies)
uv sync --all-extras

# Set up pre-commit hooks
uv run pre-commit install
```

### Testing and Code Quality

The project uses pytest for tests and several tools to maintain code quality. These checks are configured as pre-commit hooks:

```bash
# Run unit tests (excluding integration tests)
uv run pytest -v -k "not integration"

# Run integration tests (connects to external APIs)
uv run pytest -v -m integration --run-integration

# Run type checking
uv run mypy --config-file=pyproject.toml .

# Run linting to detect unused code
uv run pylint shallweswim/ tests/

# Format code with Black
uv run black .

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run with code coverage
uv run pytest --cov=shallweswim

# Run with code coverage and generate HTML report
uv run pytest --cov=shallweswim --cov-report=html
```

Note: Integration tests connect to live external APIs (NOAA CO-OPS, NOAA NDBC, USGS NWIS) and may occasionally fail if services are experiencing issues or data is temporarily unavailable.

## Monitoring & Station Outages

External data sources (NOAA CO-OPS, NOAA NDBC, USGS NWIS) occasionally experience outages. The application handles these gracefully:

- **Health check (`/api/healthy`)**: Returns 200 if at least one location can serve data. Single station outages don't mark the entire service unhealthy.
- **Status endpoint (`/api/status`)**: Returns detailed per-feed status including `is_healthy`, `is_expired`, and `age_seconds`. Use this for granular monitoring and alerting.

For production deployments, set up external monitoring on `/api/status` to alert when critical feeds become unhealthy. See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed station outage handling strategy.

### HTTP Error Codes

- **503 Service Unavailable**: External station has no data (expected, retry later)
- **500 Internal Server Error**: Bug in our code (needs immediate attention)

## API Documentation

When running locally, API documentation is available at:

- Swagger UI: http://localhost:12345/docs
- ReDoc: http://localhost:12345/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) (Center for Operational Oceanographic Products and Services) for tide, current, and temperature data
- [NOAA NDBC API](https://www.ndbc.noaa.gov/) (National Data Buoy Center) for buoy-based water temperature data
- [USGS NWIS API](https://waterservices.usgs.gov/) (National Water Information System) for water temperature and river current data
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Matplotlib](https://matplotlib.org/) for data visualization
- [Feather Icons](https://feathericons.com/) for UI icons
- [GoodService.io](https://goodservice.io/) for NYC subway information

## Continuous Integration

GitHub Actions workflows automatically verify the following on every push:

- **Unit Tests**: All unit tests pass
- **Type Checking**: No type errors are found by mypy
- **Code Formatting**: All Python code follows Black style
- **Linting**: No unused imports, variables, or arguments are detected by pylint

Additionally, a separate integration test workflow runs daily to ensure compatibility with external APIs.
