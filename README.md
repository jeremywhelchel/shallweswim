# Shall We Swim Today?

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Poetry](https://img.shields.io/badge/Poetry-Managed-blueviolet)](https://python-poetry.org/)

**A web application that helps open water swimmers make informed decisions about swim conditions.**

[shallweswim.today](https://shallweswim.today) provides real-time tide, current, and temperature data for popular open water swimming locations including:

- Coney Island / Brighton Beach (NYC)
- La Jolla Cove (San Diego)

## Features

- **Real-time conditions** from NOAA CO-OPS API
- **Tide predictions** with high/low tide times and heights
- **Current velocity** data with flood/ebb direction
- **Water temperature trends** (48-hour, 2-month, and multi-year)
- **Mobile-friendly interface** for on-the-go swimmers
- **JSON API** for programmatic access to swim conditions

## Architecture

Shall We Swim is a FastAPI application with a modular architecture:

### Core Components

- **Feed Framework (`feeds.py`)**: Modular data feed system for different data types
  - Base `Feed` class with expiration tracking and status reporting
  - Specialized feed types (NoaaTempFeed, NoaaTidesFeed, etc.) for different data sources
  - Composite feeds for combining multiple data sources
- **Data Management (`data.py`)**: Coordinates feeds and processes data from various sources
  - Manages feed lifecycle and data freshness
  - Provides status monitoring and ready-state tracking
  - Handles data processing and transformation
- **API Layer (`api.py`)**: JSON endpoints for swim conditions and status
  - Location-specific endpoints for conditions data
  - Status endpoints for monitoring system health
  - Current prediction and tide visualization endpoints
- **Web UI (`main.py`)**: HTML templates and web interface
- **NOAA CO-OPS Client (`coops.py`)**: Interacts with NOAA's Center for Operational Oceanographic Products and Services API
- **Configuration (`config.py`)**: Location settings and station IDs
- **Utilities (`util.py`)**: Common utilities for time handling and data processing

### Data Flow

1. **Data Fetching**: Specialized Feed classes fetch data (tides, currents, temperatures) from NOAA CO-OPS and other sources for configured locations
2. **Data Processing**: Raw data is processed with appropriate timezone conversions
3. **Status Monitoring**: Feed and DataManager status is tracked and exposed via API endpoints
4. **Plot Generation**: Visualizations are asynchronously generated for different time spans:
   - 48-hour tide/current predictions
   - 2-month historical temperature data
   - Multi-year temperature trends
5. **API Endpoints**: Processed data and system status are available via JSON endpoints
6. **Web UI**: Templates display the data with visualizations

## Getting Started

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- Docker (optional, for containerized deployment)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/jeremywhelchel/shallweswim.git
cd shallweswim

# Install dependencies
poetry install

# Run the development server
PORT=12345 poetry run python shallweswim/main.py
```

Then visit http://localhost:12345 in your browser.

### Run with Docker

```bash
# Build the Docker image
docker build -t shallweswim .

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

### Setup

```bash
# Install Poetry (recommended method)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up pre-commit hooks
poetry run pre-commit install
```

### Testing and Code Quality

The project uses pytest for tests and several tools to maintain code quality. These checks are configured as pre-commit hooks:

```bash
# Run unit tests (excluding integration tests)
poetry run pytest -v -k "not integration"

# Run integration tests (connects to external APIs)
poetry run pytest -v -m integration --run-integration

# Run type checking
poetry run mypy --config-file=pyproject.toml .

# Run linting to detect unused code
poetry run pylint shallweswim/ tests/

# Format code with Black
poetry run black .

# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Run with code coverage
poetry run pytest --cov=shallweswim

# Run with code coverage and generate HTML report
poetry run pytest --cov=shallweswim --cov-report=html
```

Note: Integration tests connect to live NOAA CO-OPS API and may occasionally fail if external services are experiencing issues or if the expected data is temporarily unavailable.

## API Documentation

When running locally, API documentation is available at:

- Swagger UI: http://localhost:12345/docs
- ReDoc: http://localhost:12345/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) (Center for Operational Oceanographic Products and Services) for tide, current, and temperature data
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Matplotlib](https://matplotlib.org/) for data visualization

## Continuous Integration

GitHub Actions workflows automatically verify the following on every push:

- **Unit Tests**: All unit tests pass
- **Type Checking**: No type errors are found by mypy
- **Code Formatting**: All Python code follows Black style
- **Linting**: No unused imports, variables, or arguments are detected by pylint

Additionally, a separate integration test workflow runs daily to ensure compatibility with external APIs.
