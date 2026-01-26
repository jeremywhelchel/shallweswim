#!/usr/bin/env python3
"""Generate static maps."""

import logging

from shallweswim import plot
from shallweswim.types import CurrentDirection

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Generating static files")
    for ef in [
        CurrentDirection.FLOODING.value,
        CurrentDirection.EBBING.value,
    ]:
        for magnitude_bin in plot.MAGNITUDE_BINS:
            plot.generate_and_save_current_chart(ef, magnitude_bin)
