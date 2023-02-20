#!/usr/bin/env bash

set -e;

IMAGE_TAG="us-east4-docker.pkg.dev/shallweswim/gcr/shallweswim"

gcloud config configurations activate shallweswim
gcloud builds submit --tag ${IMAGE_TAG}
gcloud run deploy --image ${IMAGE_TAG} shallweswim
