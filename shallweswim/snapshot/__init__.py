"""Published serving snapshots: model, serialization, store, and publication.

A snapshot generation is an immutable manifest referencing immutable
content-addressed objects, one Parquet object per served feed frame and one
SVG per plot, promoted through a conditionally replaced current pointer. The
rules are in DATA_PIPELINE.md, "The bundle".
"""
