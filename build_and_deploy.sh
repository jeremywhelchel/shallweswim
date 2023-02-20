#!/usr/bin/env bash

set -e;

gcloud config configurations activate shallweswim
gcloud builds submit --config cloudbuild.yaml
