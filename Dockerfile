# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
# Install production dependencies via Poetry
COPY pyproject.toml ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main
# Copy code and data
COPY shallweswim shallweswim
# Run the web service on container startup.
CMD exec uvicorn shallweswim.main:start_app --host 0.0.0.0 --port ${PORT}
