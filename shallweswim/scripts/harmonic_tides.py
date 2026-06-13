"""Compatibility imports for harmonic tide script tooling."""

from shallweswim.harmonic_tides import (
    DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR,
    DEFAULT_FIT_CONSTITUENTS,
    MODEL_SCHEMA_VERSION,
    HarmonicConstituent,
    HarmonicTideModel,
    constituent_amplitude_phase,
    fit_harmonic_model,
    load_model,
    predict_high_low_events,
)

__all__ = [
    "DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR",
    "DEFAULT_FIT_CONSTITUENTS",
    "MODEL_SCHEMA_VERSION",
    "HarmonicConstituent",
    "HarmonicTideModel",
    "constituent_amplitude_phase",
    "fit_harmonic_model",
    "load_model",
    "predict_high_low_events",
]
