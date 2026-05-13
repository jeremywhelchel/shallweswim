"""Export the FastAPI OpenAPI schema for frontend type generation."""

import json
import sys
from typing import Any

from shallweswim.main import app


def main() -> None:
    """Write the current OpenAPI schema to stdout as deterministic JSON."""
    schema: dict[str, Any] = app.openapi()
    json.dump(schema, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
