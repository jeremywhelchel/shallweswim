# Code Conventions

This document outlines the coding standards and architectural patterns for the "Shall We Swim Today?" project.

## 1. Architectural Patterns

### Modular Design
- **API (`api.py`)**: Contains *only* route handlers and request validation. Delegates business logic to `data.py` or `feeds.py`.
- **Data Management (`data.py`)**: `LocationDataManager` is the facade for all data operations per location. It coordinates multiple `Feed`s.
- **Feeds (`feeds.py`)**: Encapsulates data fetching, caching, and validation. Each data source type (e.g., `NoaaTidesFeed`) is a separate class inheriting from `Feed`.
- **Clients (`clients/`)**: Pure API clients. They handle HTTP requests, retries, and raw data parsing, but know nothing about the application's business logic (caching, expiration).

### Data Flow
`API Handler` -> `LocationDataManager` -> `Feed` -> `ApiClient` -> `External Service`

### Configuration
- All static configuration is in `config.py`.
- Use Pydantic models for configuration schemas (e.g., `LocationConfig`, `BaseFeedConfig`).
- Avoid hardcoding constants in logic files; move them to `config.py` or module-level constants if they are "magic numbers".

## 2. Coding Standards

### Typing
- **Strict Typing**: All functions and methods must have type hints.
- **Pydantic**: Use Pydantic v2 models for all data structures passed between layers (especially API responses).
  - Use `model_config = ConfigDict(...)` for model configuration.
- **Pandas**: When passing DataFrames, use `pandera` models or clear docstrings to describe the expected columns.

### Asynchronous Programming
- **FastAPI**: Route handlers must be `async`.
- **I/O Bound**: All network I/O must be asynchronous (`aiohttp`).
- **CPU Bound**: Heavy computation (like plotting with Matplotlib) must be offloaded to a process pool (`app.state.process_pool`) to avoid blocking the event loop.

### Error Handling
- **API Layer**: Catch internal exceptions and convert them to `HTTPException` with appropriate status codes (404, 503, etc.).
- **Data Layer**: Fail fast. Use `AssertionError` for "impossible" states (e.g., missing app state). Raise `ValueError` or specific custom exceptions for operational errors (e.g., parsing failure).
- **Feeds**: Feeds should handle transient network errors gracefully (e.g., retries) but expose a "healthy" status property.

### Logging
- Use the standard `logging` module.
- Format: `logging.info(f"[{context}] Message")` where context is often the location code or component name.

## 3. Testing

- **Unit Tests**: Must run fast. Mock all external network calls using `unittest.mock` or `pytest-mock`.
- **Integration Tests**: Marked with `@pytest.mark.integration`. These hit real external APIs and are run separately.
- **Fixtures**: Use `conftest.py` for shared fixtures.

## 4. Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes.
  - **Args**: List arguments and their types.
  - **Returns**: Describe the return value.
  - **Raises**: List exceptions raised.
