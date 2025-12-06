# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.13-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
# Install production dependencies via Poetry
COPY pyproject.toml ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-root
# Copy code and data
COPY shallweswim shallweswim

# Generate asset manifest for fingerprinting
RUN poetry run python -m shallweswim.scripts.generate_asset_manifest

# Run the web service on container startup with asset manifest
CMD exec python -m shallweswim.main --asset-manifest=shallweswim/static/asset-manifest.json --host 0.0.0.0 --port ${PORT}
