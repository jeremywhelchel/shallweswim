# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
# Install production dependencies.
# Note Numpy 1.24.0 is causing issues with seaborn:
# https://github.com/mwaskom/seaborn/issues/3192
# XXX: remove when 1.24.1 available
RUN pip install Flask google-cloud-logging gunicorn numpy==1.23.5 pandas scipy seaborn
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 'main:start_app()'
