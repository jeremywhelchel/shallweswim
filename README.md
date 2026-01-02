# Shall We Swim Today?

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-Managed-blueviolet)](https://github.com/astral-sh/uv)

**A web application that helps open water swimmers make informed decisions about swim conditions.**

[shallweswim.today](https://shallweswim.today) provides real-time tide, current, and temperature data for popular open water swimming locations including:

- Coney Island / Brighton Beach (NYC)
- La Jolla Cove (San Diego)
- Additional locations are being added regularly

## Features

- **Real-time conditions** from NOAA CO-OPS and USGS NWIS APIs
- **Tide predictions** with high/low tide times and heights
- **Current velocity** data with flood/ebb direction
- **Water temperature trends** (48-hour, 2-month, and multi-year)
- **Transit information** for NYC locations (subway status and alerts)
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
  - Formats temperature data with appropriate precision
- **API Layer (`api.py`)**: JSON endpoints for swim conditions and status
  - Location-specific endpoints for conditions data
  - Status endpoints for monitoring system health
  - Current prediction and tide visualization endpoints
- **Web UI (`main.py`)**: HTML templates and web interface
  - Responsive design with modern UI components
  - Conditional display of data based on feed availability
  - Transit information for NYC locations
- **NOAA CO-OPS Client (`coops.py`)**: Interacts with NOAA's Center for Operational Oceanographic Products and Services API
- **USGS NWIS Client (`nwis.py`)**: Interacts with the USGS National Water Information System API
  - Handles both Celsius and Fahrenheit temperature parameters
  - Performs necessary unit conversions
- **Configuration (`config.py`)**: Location settings and station IDs
- **Utilities (`util.py`)**: Common utilities for time handling and data processing

### Data Flow

1. **Data Fetching**: Specialized Feed classes fetch data (tides, currents, temperatures) from NOAA CO-OPS, USGS NWIS, and other sources for configured locations
2. **Data Processing**: Raw data is processed with appropriate timezone conversions and unit transformations
3. **Status Monitoring**: Feed and DataManager status is tracked and exposed via API endpoints
4. **Plot Generation**: Visualizations are asynchronously generated for different time spans:
   - 48-hour tide/current predictions
   - 2-month historical temperature data
   - Multi-year temperature trends
5. **API Endpoints**: Processed data and system status are available via JSON endpoints
6. **Web UI**: Templates display the data with visualizations and location-specific information (like transit status)

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

Note: Integration tests connect to live NOAA CO-OPS API and may occasionally fail if external services are experiencing issues or if the expected data is temporarily unavailable.

## API Documentation

When running locally, API documentation is available at:

- Swagger UI: http://localhost:12345/docs
- ReDoc: http://localhost:12345/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) (Center for Operational Oceanographic Products and Services) for tide, current, and temperature data
- [USGS NWIS API](https://waterservices.usgs.gov/) (National Water Information System) for water temperature data
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
