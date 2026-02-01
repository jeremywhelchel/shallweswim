#!/usr/bin/env bash

set -e;

gcloud builds submit --config cloudbuild.yaml
