#!/usr/bin/env bash
gcloud builds submit --tag gcr.io/shallweswim/shallweswim && \
gcloud run deploy --image gcr.io/shallweswim/shallweswim --platform managed shallweswim
