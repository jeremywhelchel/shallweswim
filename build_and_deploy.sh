#!/usr/bin/env bash
gcloud builds submit --tag gcr.io/watertemp/watertemp
gcloud run deploy --image gcr.io/watertemp/watertemp --platform managed watertemp
