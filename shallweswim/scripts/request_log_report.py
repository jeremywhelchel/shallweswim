"""Summarize Cloud Run request logs without collecting application analytics."""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from shallweswim.config.locations import CONFIGS

BOT_PATTERN = re.compile(
    r"bot|crawler|spider|slurp|headless|facebookexternalhit|preview", re.IGNORECASE
)
STATIC_PATTERN = re.compile(
    r"\.(?:css|ico|jpe?g|js|map|png|svg|woff2?)$", re.IGNORECASE
)
UPTIME_USER_AGENT = "GoogleStackdriverMonitoring-UptimeChecks"


@dataclass(frozen=True)
class RequestEvent:
    timestamp: str
    status: str
    path: str
    host: str
    user_agent: str
    remote_ip: str

    @property
    def traffic_class(self) -> str:
        return classify_user_agent(self.user_agent)

    @property
    def location(self) -> str | None:
        segments = [segment for segment in self.path.split("/") if segment]
        if segments and segments[0] in CONFIGS:
            return segments[0]
        if len(segments) >= 2 and segments[0] == "api" and segments[1] in CONFIGS:
            return segments[1]
        return None


def classify_user_agent(user_agent: str) -> str:
    """Classify a user agent into a deliberately small reporting category."""
    if UPTIME_USER_AGENT in user_agent:
        return "monitor"
    if BOT_PATTERN.search(user_agent):
        return "bot"
    if user_agent.lower().startswith("curl/"):
        return "curl"
    if "mozilla/" in user_agent.lower():
        return "browser"
    return "other"


def operation_for_path(path: str) -> str:
    """Return a bounded operation name for a request path."""
    segments = [segment for segment in path.split("/") if segment]
    if path == "/":
        return "homepage"
    if path == "/locations":
        return "locations-page"
    if segments and segments[0] in CONFIGS:
        return "location-page" if len(segments) == 1 else "location-subpage"
    if len(segments) >= 2 and segments[0] == "api" and segments[1] in CONFIGS:
        suffix = "/".join(segments[2:4]) or "root"
        return f"api/{suffix}"
    if path.startswith(("/assets/", "/static/")) or STATIC_PATTERN.search(path):
        return "asset"
    if path.startswith("/api/"):
        return "site-api"
    if path in {"/manifest.json", "/robots.txt", "/sitemap.xml"}:
        return "site-metadata"
    return "unknown/probe"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and summarize automatic Cloud Run HTTP request logs."
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--days", type=int, default=7, help="Look back N days (default: 7)."
    )
    window.add_argument("--start", help="Inclusive ISO-8601 start timestamp or date.")
    parser.add_argument("--end", help="Exclusive ISO-8601 end timestamp or date.")
    parser.add_argument(
        "--project", help="GCP project; defaults to gcloud configuration."
    )
    parser.add_argument(
        "--service", default="shallweswim", help="Cloud Run service name."
    )
    parser.add_argument("--host", help="Only include this exact request host.")
    parser.add_argument(
        "--location", choices=sorted(CONFIGS), help="Only this location."
    )
    parser.add_argument(
        "--traffic",
        choices=("all", "browser", "bot", "curl", "monitor", "other"),
        default="all",
        help="Only one traffic class (default: all).",
    )
    parser.add_argument(
        "--include-uptime",
        action="store_true",
        help="Include managed uptime-check traffic (excluded by default).",
    )
    parser.add_argument("--limit", type=int, default=100_000, help="Maximum log rows.")
    parser.add_argument("--top", type=int, default=15, help="Rows per ranked section.")
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Inactivity gap that starts a new estimated visit (default: 30).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args(argv)
    if args.host:
        args.host = args.host.lower()
    return args


def _timestamp(value: str) -> str:
    if "T" not in value:
        return f"{value}T00:00:00Z"
    if value.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", value):
        return value
    value = f"{value}Z"
    return value


def _validated_filter_value(value: str, *, name: str, pattern: str) -> str:
    if not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def build_filter(args: argparse.Namespace) -> str:
    """Build a Cloud Logging filter from validated command-line values."""
    if args.limit <= 0 or (args.days is not None and args.days <= 0):
        raise ValueError("--limit and --days must be positive")
    if args.start:
        start = _timestamp(args.start)
    else:
        start = (dt.datetime.now(dt.UTC) - dt.timedelta(days=args.days)).isoformat()

    service = _validated_filter_value(
        args.service, name="service", pattern=r"[A-Za-z0-9-]+"
    )
    parts = [
        'resource.type="cloud_run_revision"',
        f'resource.labels.service_name="{service}"',
        'logName:"run.googleapis.com%2Frequests"',
        f'timestamp>="{start}"',
    ]
    if args.end:
        parts.append(f'timestamp<"{_timestamp(args.end)}"')
    if not args.include_uptime and args.traffic != "monitor":
        parts.append(f'NOT httpRequest.userAgent:"{UPTIME_USER_AGENT}"')
    if args.traffic == "browser":
        parts.append('httpRequest.userAgent:"Mozilla/"')
    elif args.traffic == "curl":
        parts.append('httpRequest.userAgent:"curl/"')
    elif args.traffic == "monitor":
        parts.append(f'httpRequest.userAgent:"{UPTIME_USER_AGENT}"')
    if args.host:
        host = _validated_filter_value(
            args.host.lower(), name="host", pattern=r"[A-Za-z0-9.-]+"
        )
        parts.append(f'httpRequest.requestUrl:"//{host}/"')
    if args.location:
        parts.append(
            f'(httpRequest.requestUrl:"/{args.location}" OR '
            f'httpRequest.requestUrl:"/api/{args.location}/")'
        )
    return " AND ".join(parts)


def read_events(args: argparse.Namespace) -> tuple[list[RequestEvent], int]:
    """Run one read-only gcloud query and parse its request events."""
    command = ["gcloud"]
    if args.project:
        command.extend(("--project", args.project))
    command.extend(
        (
            "logging",
            "read",
            build_filter(args),
            f"--limit={args.limit}",
            "--order=desc",
            "--format=json",
        )
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or str(error)
        raise RuntimeError(f"gcloud logging read failed:\n{detail}") from error
    rows: list[dict[str, Any]] = json.loads(result.stdout)
    events = []
    for row in rows:
        request = row.get("httpRequest", {})
        url = urlsplit(request.get("requestUrl", ""))
        event = RequestEvent(
            timestamp=row.get("timestamp", ""),
            status=str(request.get("status", "?")),
            path=url.path or "/",
            host=url.hostname or "",
            user_agent=request.get("userAgent", ""),
            remote_ip=request.get("remoteIp", ""),
        )
        if args.traffic != "all" and event.traffic_class != args.traffic:
            continue
        if args.host and event.host != args.host:
            continue
        if args.location and event.location != args.location:
            continue
        events.append(event)
    return events, len(rows)


def summarize(
    events: list[RequestEvent],
    *,
    fetched_rows: int,
    limit: int,
    top: int,
    session_gap_minutes: int = 30,
) -> dict[str, Any]:
    """Build a privacy-conscious aggregate report; never return raw IP addresses."""
    counter_names = (
        "traffic",
        "status",
        "location",
        "operation",
        "path",
        "day",
        "host",
    )
    counters = {name: collections.Counter[str]() for name in counter_names}
    unique_ips: set[str] = set()
    browser_ips: set[str] = set()
    for event in events:
        traffic_class = event.traffic_class
        counters["traffic"][traffic_class] += 1
        counters["status"][event.status] += 1
        counters["path"][event.path] += 1
        counters["operation"][operation_for_path(event.path)] += 1
        counters["day"][event.timestamp[:10]] += 1
        counters["host"][event.host] += 1
        if event.location:
            counters["location"][event.location] += 1
        if event.remote_ip:
            unique_ips.add(event.remote_ip)
            if traffic_class == "browser":
                browser_ips.add(event.remote_ip)

    timestamps = [event.timestamp for event in events if event.timestamp]
    session_summary = estimate_sessions(events, gap_minutes=session_gap_minutes)
    return {
        "matched_events": len(events),
        "fetched_rows": fetched_rows,
        "limit": limit,
        "truncated": fetched_rows >= limit,
        "newest": max(timestamps, default=None),
        "oldest": min(timestamps, default=None),
        "approx_unique_ips": len(unique_ips),
        "approx_unique_browser_ips": len(browser_ips),
        **session_summary,
        **{
            name: sorted(counter.items()) if name == "day" else counter.most_common(top)
            for name, counter in counters.items()
        },
    }


def _event_time(timestamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _is_meaningful_browser_event(event: RequestEvent) -> bool:
    if event.traffic_class != "browser" or not event.remote_ip or not event.user_agent:
        return False
    try:
        status = int(event.status)
    except ValueError:
        return False
    operation = operation_for_path(event.path)
    return 200 <= status < 400 and (
        operation
        in {
            "homepage",
            "locations-page",
            "location-page",
            "location-subpage",
        }
        or operation.startswith("api/")
    )


def estimate_sessions(
    events: list[RequestEvent], *, gap_minutes: int = 30
) -> dict[str, Any]:
    """Estimate visits using ephemeral IP/user-agent identities and inactivity gaps."""
    if gap_minutes <= 0:
        raise ValueError("session gap must be positive")
    salt = os.urandom(16)
    meaningful = [event for event in events if _is_meaningful_browser_event(event)]
    meaningful.sort(key=lambda event: event.timestamp)
    last_seen: dict[bytes, dt.datetime] = {}
    current_session: dict[bytes, int] = {}
    devices: set[bytes] = set()
    session_locations: set[tuple[int, str]] = set()
    location_visits: collections.Counter[str] = collections.Counter()
    sessions = 0
    document_entries = 0
    gap = dt.timedelta(minutes=gap_minutes)

    for event in meaningful:
        identity = hashlib.sha256(
            salt + event.remote_ip.encode() + b"\0" + event.user_agent.encode()
        ).digest()
        event_time = _event_time(event.timestamp)
        devices.add(identity)
        if identity not in last_seen or event_time - last_seen[identity] > gap:
            sessions += 1
            current_session[identity] = sessions
        last_seen[identity] = event_time

        operation = operation_for_path(event.path)
        if operation in {
            "homepage",
            "locations-page",
            "location-page",
            "location-subpage",
        }:
            document_entries += 1
        if event.location:
            session_location = (current_session[identity], event.location)
            if session_location not in session_locations:
                session_locations.add(session_location)
                location_visits[event.location] += 1

    return {
        "estimated_sessions": sessions,
        "estimated_devices": len(devices),
        "session_gap_minutes": gap_minutes,
        "sessionized_requests": len(meaningful),
        "document_entries": document_entries,
        "estimated_location_sessions": location_visits.most_common(),
    }


def print_text(report: dict[str, Any]) -> None:
    """Print a compact operator-facing report."""
    print(
        f"Coverage: {report['oldest']} to {report['newest']} "
        f"({report['matched_events']:,} matched; {report['fetched_rows']:,} fetched)"
    )
    if report["truncated"]:
        print("WARNING: row limit reached; the requested time window is not complete.")
    print(
        "Approximate unique IPs: "
        f"{report['approx_unique_ips']:,} total, "
        f"{report['approx_unique_browser_ips']:,} browser"
    )
    print(
        f"Estimated visits ({report['session_gap_minutes']}-minute gap): "
        f"{report['estimated_sessions']:,} across "
        f"{report['estimated_devices']:,} ephemeral browser identities"
    )
    print(
        f"Document entries: {report['document_entries']:,}; "
        f"requests used for sessionization: {report['sessionized_requests']:,}"
    )
    print("\nEstimated location visits:")
    for label, count in report["estimated_location_sessions"]:
        print(f"  {count:>8,}  {label}")
    for name in ("traffic", "location", "operation", "status", "path", "day", "host"):
        print(f"\n{name.title()}:")
        for label, count in report[name]:
            print(f"  {count:>8,}  {label or '(empty)'}")


def main() -> None:
    args = _parse_args()
    try:
        events, fetched_rows = read_events(args)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    report = summarize(
        events,
        fetched_rows=fetched_rows,
        limit=args.limit,
        top=args.top,
        session_gap_minutes=args.session_gap_minutes,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print_text(report)


if __name__ == "__main__":
    main()
