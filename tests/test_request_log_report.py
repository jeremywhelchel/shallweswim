import argparse

import pytest

from shallweswim.scripts.request_log_report import (
    RequestEvent,
    _timestamp,
    build_filter,
    classify_user_agent,
    estimate_sessions,
    operation_for_path,
    summarize,
)


def _filter_args(**overrides: object) -> argparse.Namespace:
    values = {
        "limit": 100,
        "days": None,
        "start": "2026-08-01",
        "end": "2026-08-02T12:00:00",
        "service": "shallweswim",
        "include_uptime": False,
        "traffic": "all",
        "host": None,
        "location": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_classify_user_agent_uses_bounded_categories() -> None:
    assert classify_user_agent("Mozilla/5.0 Safari/537.36") == "browser"
    assert classify_user_agent("curl/8.0") == "curl"
    assert classify_user_agent("Googlebot/2.1") == "bot"
    assert classify_user_agent("custom-client") == "other"


def test_timestamp_treats_naive_dates_and_datetimes_as_utc() -> None:
    assert _timestamp("2026-08-01") == "2026-08-01T00:00:00Z"
    assert _timestamp("2026-08-01T12:00:00") == "2026-08-01T12:00:00Z"
    assert _timestamp("2026-08-01T12:00:00-04:00").endswith("-04:00")


def test_filter_excludes_managed_uptime_by_default_without_hiding_health_path() -> None:
    log_filter = build_filter(_filter_args())

    assert 'timestamp>="2026-08-01T00:00:00Z"' in log_filter
    assert 'timestamp<"2026-08-02T12:00:00Z"' in log_filter
    assert "NOT httpRequest.userAgent" in log_filter
    assert "/api/healthy" not in log_filter


def test_monitor_filter_includes_only_managed_uptime() -> None:
    log_filter = build_filter(_filter_args(traffic="monitor"))

    assert (
        'httpRequest.userAgent:"GoogleStackdriverMonitoring-UptimeChecks"' in log_filter
    )
    assert "NOT httpRequest.userAgent" not in log_filter


def test_filter_rejects_values_that_could_break_filter_quoting() -> None:
    with pytest.raises(ValueError, match="Invalid service"):
        build_filter(_filter_args(service='bad"service'))


def test_location_and_operation_are_inferred_from_page_and_api_paths() -> None:
    page = RequestEvent("2026-08-23T12:00:00Z", "200", "/nyc", "example", "", "")
    api = RequestEvent(
        "2026-08-23T12:00:01Z",
        "200",
        "/api/sfo/plots/live_temps",
        "example",
        "",
        "",
    )
    assert page.location == "nyc"
    assert operation_for_path(page.path) == "location-page"
    assert api.location == "sfo"
    assert operation_for_path(api.path) == "api/plots/live_temps"


def test_non_location_operations_separate_assets_metadata_and_unknown_probes() -> None:
    assert operation_for_path("/assets/app.js") == "asset"
    assert operation_for_path("/static/favicon.png") == "asset"
    assert operation_for_path("/favicon.ico") == "asset"
    assert operation_for_path("/api/app/bootstrap") == "site-api"
    assert operation_for_path("/api/status") == "site-api"
    assert operation_for_path("/manifest.json") == "site-metadata"
    assert operation_for_path("/robots.txt") == "site-metadata"
    assert operation_for_path("/wp-login.php") == "unknown/probe"


def test_summary_counts_ips_without_exposing_them() -> None:
    events = [
        RequestEvent(
            "2026-08-22T12:00:00Z",
            "200",
            "/nyc",
            "prod",
            "Mozilla/5.0",
            "1.2.3.4",
        ),
        RequestEvent(
            "2026-08-23T12:00:00Z",
            "200",
            "/api/nyc/conditions",
            "prod",
            "curl/8",
            "1.2.3.4",
        ),
    ]

    report = summarize(events, fetched_rows=2, limit=100, top=10)

    assert report["approx_unique_ips"] == 1
    assert report["approx_unique_browser_ips"] == 1
    assert report["location"] == [("nyc", 2)]
    assert "1.2.3.4" not in str(report)


def test_summary_lists_all_days_chronologically() -> None:
    events = [
        RequestEvent(f"2026-08-{day:02d}T12:00:00Z", "200", "/", "prod", "curl/8", "1")
        for day in range(1, 18)
    ]

    report = summarize(events, fetched_rows=len(events), limit=100, top=2)

    assert report["day"] == [(f"2026-08-{day:02d}", 1) for day in range(1, 18)]


def test_estimated_sessions_ignore_polling_until_inactivity_gap() -> None:
    events = [
        RequestEvent(
            timestamp,
            "200",
            path,
            "prod",
            "Mozilla/5.0",
            "1.2.3.4",
        )
        for timestamp, path in (
            ("2026-08-23T12:00:00Z", "/nyc"),
            ("2026-08-23T12:05:00Z", "/api/nyc/conditions"),
            ("2026-08-23T12:10:00Z", "/api/nyc/conditions"),
            ("2026-08-23T13:00:00Z", "/api/nyc/conditions"),
        )
    ]

    result = estimate_sessions(events, gap_minutes=30)

    assert result["estimated_sessions"] == 2
    assert result["estimated_devices"] == 1
    assert result["document_entries"] == 1
    assert result["estimated_location_sessions"] == [("nyc", 2)]


def test_estimated_sessions_exclude_assets_errors_and_non_browser_requests() -> None:
    events = [
        RequestEvent(
            "2026-08-23T12:00:00Z", "200", "/static/app.js", "prod", "Mozilla/5.0", "1"
        ),
        RequestEvent("2026-08-23T12:01:00Z", "404", "/nyc", "prod", "Mozilla/5.0", "1"),
        RequestEvent("2026-08-23T12:02:00Z", "200", "/nyc", "prod", "curl/8", "1"),
    ]

    result = estimate_sessions(events)

    assert result["estimated_sessions"] == 0
