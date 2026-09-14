"""Clone-and-run local entry point: the job and the web app in one process.

``uv run python -m shallweswim.local`` needs no bucket and no credentials. It
points the three store variables (``SHALLWESWIM_ARCHIVE_BUCKET``,
``SHALLWESWIM_ARCHIVE_READ_BUCKET``, ``SHALLWESWIM_SNAPSHOT_READ_BUCKET``) at
one local store locator, runs the capture job's publishing cycle against it on
a timer, and serves the web app from the same process, which loads the
generations that cycle publishes exactly as the deployed service loads the
job's. ``--store-dir PATH`` keeps the archive and the published generations on
disk, so a second start hydrates history from the archive instead of refetching
every configured year; the default ``memory`` locator keeps them in this
process only.

The app itself is unchanged: ``main.start_app`` builds it, and the updater is
composed around its lifespan rather than conditionally inside it. This module
wraps ``app.router.lifespan_context``, so the app's own startup runs first and
the updater task starts after it, with the HTTP session and process pool the
app created already in ``app.state``; the wrapper cancels the task before the
app's shutdown runs. ``main.py`` therefore has no notion of local mode.

Uvicorn runs the application object in this process rather than the factory
string, because the store, the published generations, and the loaded bundle all
live in this process's memory. ``--reload`` is unsupported for the same reason.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator

import fastapi
import uvicorn

from shallweswim import capture
from shallweswim import main as main_module
from shallweswim.archive.store import MEMORY_LOCATOR
from shallweswim.clients import create_api_clients
from shallweswim.logging_utils import setup_logging
from shallweswim.snapshot.store import SNAPSHOT_READ_BUCKET_ENV_VAR

# The cadence the design targets for the production job, and the interval the
# published generation's freshness budget is written against.
DEFAULT_CADENCE_MINUTES = 10

# Every store the process reads or writes is the same local store: the cycle's
# archive writes, the historical feed's hydration reads, and the web half's
# published generations.
STORE_ENV_VARS = (
    capture.ARCHIVE_BUCKET_ENV_VAR,
    capture.ARCHIVE_READ_BUCKET_ENV_VAR,
    SNAPSHOT_READ_BUCKET_ENV_VAR,
)


def apply_store_locator(locator: str) -> None:
    """Point every store variable at this run's locator, overriding the shell.

    The operator's `.env` may name a real bucket; a local run must never reach
    it, so the variables are set unconditionally rather than filled in when
    absent.

    Args:
        locator: The store locator, as `archive.store.object_store` resolves
            it: `memory`, or a path containing a separator.
    """
    for name in STORE_ENV_VARS:
        os.environ[name] = locator


async def run_cycles(
    app: fastapi.FastAPI, *, locator: str, cadence_seconds: float
) -> None:
    """Run the job's publishing cycle immediately, then every cadence.

    The cycle is `capture.publish_locations`, unchanged: every location's full
    serving cycle, archive capture inside each feed update, plots, and one
    published generation. It runs in the app's process pool and over the app's
    HTTP session, so this process holds one of each.

    A cycle that raises is logged at ERROR and the loop continues at the next
    cadence; a cycle that overruns the cadence starts the next one immediately.

    Args:
        app: The running application, whose state holds the process pool and
            the HTTP session the cycle uses.
        locator: Store locator the generation is published into.
        cadence_seconds: Seconds between the start of one cycle and the next.
    """
    while True:
        started_at = time.monotonic()
        logging.info(f"[local] publishing cycle starting, store {locator}")
        try:
            await capture.publish_locations(
                create_api_clients(app.state.http_session),
                uuid.uuid4().hex,
                pool=app.state.process_pool,
                locator=locator,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # publish_locations isolates feed and publication failures itself,
            # so reaching here means the cycle as a whole broke.
            logging.error(f"[local] publishing cycle failed: {error}")
        await asyncio.sleep(max(0.0, cadence_seconds - (time.monotonic() - started_at)))


def install_updater(
    app: fastapi.FastAPI, *, locator: str, cadence_seconds: float
) -> None:
    """Wrap the app's lifespan so the updater runs alongside it.

    The app's own lifespan is entered first and exited last, so the updater
    starts only once the process pool and HTTP session exist and is cancelled
    before they are torn down.

    Args:
        app: The application to wrap; its lifespan is replaced by the wrapper.
        locator: Store locator the cycles publish into.
        cadence_seconds: Seconds between the start of one cycle and the next.
    """
    app_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan_with_updater(app: fastapi.FastAPI) -> AsyncGenerator[None]:
        async with app_lifespan(app):
            task = asyncio.create_task(
                run_cycles(app, locator=locator, cadence_seconds=cadence_seconds)
            )
            # Visible state, so a stuck or stopped updater can be inspected.
            app.state.local_updater = task
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = lifespan_with_updater


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the local entry point's arguments.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m shallweswim.local",
        description=(
            "Run the publishing job and the web app in one process against a "
            "local store."
        ),
    )
    parser.add_argument(
        "--asset-manifest",
        type=str,
        help="Path to asset manifest file for fingerprint-based cache busting",
    )
    parser.add_argument(
        "--frontend-dist",
        type=str,
        default=str(main_module.DEFAULT_FRONTEND_DIST),
        help="Path to built frontend app output directory",
    )
    parser.add_argument(
        "--require-frontend-dist",
        action="store_true",
        help="Fail startup if the built frontend app shell is missing",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8080)),
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=None,
        help=(
            "Directory holding the archive and the published generations. "
            "Defaults to an in-process memory store, which starts empty every "
            "run; a directory persists both across restarts, so the next start "
            "hydrates historical years from it instead of refetching them."
        ),
    )
    parser.add_argument(
        "--cadence",
        type=float,
        default=DEFAULT_CADENCE_MINUTES,
        help="Minutes between publishing cycles (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the job cycle and the web app in this process.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    log_format = setup_logging()
    # An absolute path always contains a separator, so a bare directory name
    # such as `store` can never be read as a bucket name.
    locator = os.path.abspath(args.store_dir) if args.store_dir else MEMORY_LOCATOR
    apply_store_locator(locator)
    logging.info(
        f"[local] store {locator}, publishing every {args.cadence} minutes, "
        f"serving on {args.host}:{args.port}"
    )

    app = main_module.start_app(
        asset_manifest=args.asset_manifest,
        frontend_dist=args.frontend_dist,
        require_frontend_dist=args.require_frontend_dist,
    )
    install_updater(app, locator=locator, cadence_seconds=args.cadence * 60)

    # The application object, not the factory string: the store and the loaded
    # generation live in this process and cannot survive a reload child.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=log_format == "console",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
