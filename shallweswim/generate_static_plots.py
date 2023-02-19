#!/usr/bin/env python3
"""Generate static maps."""

import data
import logging


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Generating static files")
    for ef in ["flooding", "ebbing"]:
        for magnitude_bin in data.MAGNITUDE_BINS:
            data.GenerateCurrentChart(ef, magnitude_bin)
