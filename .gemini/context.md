# Shall We Swim Today? Project Context

This document provides context for AI coding agents to understand the "Shall We Swim Today?" project.

## Quick Links

- **[Code Conventions](docs/CONVENTIONS.md)**: STRICTLY follow these standards for all code changes.
- **[Project README](../README.md)**: General project overview and setup.

## Project Overview

"Shall We Swim Today?" is a FastAPI web application that provides real-time tide, current, and temperature data for open water swimming locations. It fetches data from NOAA CO-OPS and USGS NWIS APIs.

**Tech Stack:**

- **Language**: Python 3.13
- **Web Framework**: FastAPI
- **Dependency Management**: uv
- **Data Processing**: Pandas, NumPy
- **Plotting**: Matplotlib (run in process pool)
- **Validation**: Pydantic v2
- **Testing**: pytest

## Codebase Map

### Core Logic (`shallweswim/`)

- **`main.py`**: Application entry point. Handles `lifespan` (startup/shutdown) and serves HTML.
- **`api.py`**: JSON API endpoints. _Entry point for all data requests._
- **`data.py`**: **Critical Component.** `LocationDataManager` orchestrates data fetching for a specific location. It manages multiple `Feed` instances.
- **`feeds.py`**: **Critical Component.** The `Feed` abstract base class and its implementations. Handles caching, expiration, and data readiness.
- **`config.py`**: Central configuration registry. Defines known swimming locations and their data sources.
- **`plot.py`**: Generates static plots (tide charts, temperature trends). _Note: CPU-intensive, usually run in a separate process._
- **`util.py`**: Shared utilities (timezones, date formatting).

### External Interactions (`shallweswim/clients/`)

- **`base.py`**: Base API client with retry logic.
- **`coops.py`, `nwis.py`, `ndbc.py`**: Specific clients for NOAA/USGS/NDBC services.

### Data Models

- **`api_types.py`**: Pydantic models for API responses.
- **`dataframe_models.py`**: Pandera schemas for validation of internal DataFrames.

## Common Development Tasks

### 1. Adding a New Data Feed Type

1.  **Define Client**: If needed, add a client in `shallweswim/clients/`.
2.  **Create Feed Class**: In `shallweswim/feeds.py`, create a class inheriting from `Feed` (or a subclass like `NoaaBaseFeed`).
3.  **Implement `fetch_data`**: Implement the method to call the client and return a DataFrame.
4.  **Register Config**: Add a configuration model in `config.py`.

### 2. Adding a New Swimming Location

1.  **Find Station IDs**: Locate the NOAA/USGS station IDs for tides, currents, and temp.
2.  **Update Config**: Add a new `LocationConfig` entry in `shallweswim/config.py`.
3.  **Verify**: Run the app locally and check the new endpoint `/api/{new_loc}/conditions`.

### 3. Modifying the API

1.  **Update Model**: Change `shallweswim/api_types.py` to reflect the new response structure.
2.  **Update Logic**: Modify `shallweswim/api.py` to populate the new fields.
3.  **Test**: Update `tests/test_api.py`.

## Key Concepts to Remember

- **Process Pool**: Plotting is slow. `main.py` creates a `ProcessPoolExecutor` in `app.state.process_pool`. Use it for `plot.py` functions.
- **Data Freshness**: Feeds self-manage expiration. `api.py` just asks for data; the feed decides if it needs to fetch fresh data or return cached data.
- **Timezones**: All internal times should be UTC-aware or explicitly handled. Use `util.effective_time` for shifting times relative to a location's timezone.
