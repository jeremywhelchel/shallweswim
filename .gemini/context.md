# Shall We Swim Today? Project Context

This document provides context for AI coding agents to understand the "Shall We Swim Today?" project.

## Project Overview

"Shall We Swim Today?" is a FastAPI web application that provides real-time tide, current, and temperature data for open water swimming locations. It helps swimmers make informed decisions about swim conditions.

The application is built with Python 3.13, FastAPI, and Poetry for dependency management. It fetches data from NOAA CO-OPS and USGS NWIS APIs.

## Key Files and Directories

- `shallweswim/`: The main application package.
  - `main.py`: The FastAPI application entry point. It handles application startup, lifespan events, and serves HTML templates.
  - `api.py`: Defines the API endpoints for fetching swim conditions, status, and data. It contains the core logic for handling API requests.
  - `data.py`: Manages data feeds and coordinates data fetching from various sources. The `LocationDataManager` class is central to this process.
  - `config.py`: Contains the configuration for all supported swimming locations, including station IDs and data source details. This is the first place to look when adding a new location or modifying an existing one.
  - `feeds.py`: A modular data feed framework for different data types (e.g., tides, temperatures).
  - `clients/`: Contains the API clients for interacting with external services like NOAA CO-OPS (`coops.py`), USGS NWIS (`nwis.py`), and NDBC (`ndbc.py`).
  - `plot.py`: Handles the generation of plots and visualizations.
  - `static/`: Static assets like CSS, JavaScript, and images.
  - `templates/`: Jinja2 HTML templates for the web UI.
- `tests/`: Contains the test suite.
  - `test_api.py`: Tests for the API endpoints.
  - `test_data.py`: Tests for the data management logic.
  - Integration tests are marked with the `integration` marker and connect to external APIs.
- `pyproject.toml`: Defines project dependencies, scripts, and tool configurations (mypy, pylint, pytest).
- `Dockerfile`: For building and running the application in a Docker container.
- `build_and_deploy.sh`: Script for deploying the application to Google Cloud Run.

## Detailed Data Flow

A typical request for swim conditions for a location (e.g., `/api/nyc/conditions`) follows this path:

1.  **`main.py` receives the request:** The FastAPI application routes the request to the appropriate handler in `api.py`.
2.  **`api.py` handles the request:** The `location_conditions` function in `api.py` is called.
3.  **Validate Location:** The function first validates the location code using `config.get()`.
4.  **Retrieve Data Manager:** It retrieves the `LocationDataManager` instance for the requested location from the `app.state.data_managers` dictionary.
5.  **Fetch Data:** The `LocationDataManager`'s methods (e.g., `get_current_temperature()`, `get_current_tide_info()`) are called.
6.  **Access Feeds:** These methods, in turn, access the underlying data `Feed` objects (e.g., `NoaaTempFeed`, `NoaaTidesFeed`) which are managed by the `LocationDataManager`.
7.  **Fetch from External APIs:** If the data in a feed is expired, the feed will use one of the API clients from the `clients/` directory (e.g., `CoopsApi`) to fetch fresh data from the external NOAA or USGS APIs.
8.  **Process and Return Data:** The raw data is processed, validated, and transformed into Pydantic models defined in `api_types.py`.
9.  **JSON Response:** The Pydantic models are serialized to a JSON response and sent back to the client.

## Core Abstractions

- **`LocationDataManager` (`data.py`):** This is the central class for managing all data related to a single location. It is responsible for:
  - Initializing and managing all data `Feed`s for a location.
  - Providing a high-level API for accessing processed data (e.g., current temperature, tide info).
  - Tracking the overall status and health of the data for a location.
- **`Feed` Framework (`feeds.py`):** This is a modular system for fetching and managing data from different sources.
  - Each `Feed` is responsible for a specific type of data (e.g., `NoaaTidesFeed`, `NwisTempFeed`).
  - Feeds handle data fetching, caching (with expiration), and validation.
  - This makes it easy to add new data sources without modifying the core data management logic.

## Configuration Schema (`config.py`)

- The application's configuration is defined using Pydantic models for type safety and validation.
- `LocationConfig`: The main model that defines a swimming location.
- `BaseFeedConfig` and its subclasses (e.g., `CoopsTempFeedConfig`, `NwisCurrentFeedConfig`): These models define the specific parameters for each data source, such as station IDs and API endpoints. This structured approach makes the configuration clear, explicit, and less prone to errors.

## API Data Models (`api_types.py`)

- The API uses Pydantic models to define the structure of its JSON responses.
- Key models include:
  - `LocationConditions`: The main response model for the `/api/{location}/conditions` endpoint.
  - `TemperatureInfo`, `TideInfo`, `CurrentInfo`: Sub-models that structure the different types of data.
- Using these models ensures that the API responses are consistent and well-documented.

## Testing Strategy

- The project uses `pytest` for testing.
- **Unit Tests:** These tests are designed to run quickly and do not make external API calls. They use mock objects (created with `unittest.mock`) to simulate the behavior of external services. The `mock_data_managers` fixture in `tests/test_api.py` is a good example of this approach.
- **Integration Tests:** These tests are marked with the `integration` pytest marker and are designed to test the application's integration with external APIs (NOAA, USGS). They are run separately from the unit tests.

## How to Add a New Location

1.  **Add a new `LocationConfig` object** to the `_CONFIG_LIST` in `shallweswim/config.py`.
2.  **Provide the necessary data sources** (temperature, tides, currents) for the new location. You will likely need to find the correct station IDs from NOAA CO-OPS or USGS NWIS.
3.  **Enable the location** by setting `enabled=True`.
4.  **Add tests** for the new location.
