# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.13-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Install production dependencies via uv (without installing the project itself yet)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
# Copy code and data
COPY shallweswim shallweswim
# Now install the project itself
RUN uv sync --frozen --no-dev

# Generate asset manifest for fingerprinting
RUN uv run python -m shallweswim.scripts.generate_asset_manifest

# Run the web service on container startup with asset manifest
CMD exec uv run python -m shallweswim.main --asset-manifest=shallweswim/static/asset-manifest.json --host 0.0.0.0 --port ${PORT}
