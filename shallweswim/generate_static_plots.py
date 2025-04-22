#!/usr/bin/env python3
"""Generate static maps."""

import logging
from shallweswim import plot


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Generating static files")
    for ef in ["flooding", "ebbing"]:
        for magnitude_bin in plot.MAGNITUDE_BINS:
            plot.generate_current_chart(ef, magnitude_bin)
