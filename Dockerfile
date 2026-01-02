# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.13-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True
# Copy local code to the container image.
ENV APP_HOME=/app
WORKDIR $APP_HOME
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the cache path to a consistent location for cache mounting
ENV UV_CACHE_DIR=/opt/uv-cache/

# Add the virtual environment to PATH
# This removes the need to type 'uv run' repeatedly and allows standard tools to work normally.
ENV VIRTUAL_ENV=$APP_HOME/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install production dependencies via uv (without installing the project itself yet)
COPY pyproject.toml uv.lock README.md ./
# Use cache mount and compile bytecode for faster builds and startup
RUN --mount=type=cache,target=/opt/uv-cache/ \
    uv sync --frozen --no-dev --no-install-project --compile-bytecode

# Copy code and data
COPY shallweswim shallweswim
# Now install the project itself
RUN --mount=type=cache,target=/opt/uv-cache/ \
    uv sync --frozen --no-dev --compile-bytecode

# Generate asset manifest for fingerprinting
RUN python -m shallweswim.scripts.generate_asset_manifest

# Run the web service on container startup with asset manifest
CMD ["python", "-m", "shallweswim.main", "--asset-manifest=shallweswim/static/asset-manifest.json", "--host", "0.0.0.0", "--port", "8080"]
