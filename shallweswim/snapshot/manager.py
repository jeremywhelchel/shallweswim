"""Read-only serving state for one location, loaded from a published generation.

`SnapshotLocationManager` answers the same route-facing calls as
`LocationDataManager` (both satisfy `core.serving.LocationServing`), but holds
frames, plot bytes, and manifest metadata from a loaded generation instead of
feeds that fetch. It is immutable: the derived tide and current prediction
frames are computed once at construction, and a new generation constructs new
managers rather than mutating these. Its status applies the feed scheduling
and health rules to the manifest's timestamps, so bundle-served data reports
age, expiry, and health exactly as fetched data does.

The query wrappers duplicate the one-line delegations in
`LocationDataManager` on purpose. They are transitional: when the web service
serves only from generations, the fetching manager leaves the web runtime
path and this class is the only implementation the routes see.
"""

import datetime
import logging

import pandas as pd

from shallweswim import api_types
from shallweswim import config as config_lib
from shallweswim.core import queries
from shallweswim.core.feeds import HEALTH_CHECK_BUFFER, FeedName, PlotName
from shallweswim.snapshot.model import FeedMetadata, FeedSnapshot, LocationManifest
from shallweswim.types import (
    CurrentInfo,
    DataSourceType,
    LegacyChartInfo,
    TemperatureReading,
    TideInfo,
    TideState,
)
from shallweswim.util import summarize_dataframe, utc_now


def _naive_utc(timestamp: datetime.datetime) -> datetime.datetime:
    """Return a manifest instant as the naive UTC clock the status API uses."""
    return timestamp.astimezone(datetime.UTC).replace(tzinfo=None)


def feed_status_from_manifest(
    feed_name: FeedName,
    location_code: str,
    feed: FeedMetadata,
    frame: pd.DataFrame,
    now: datetime.datetime,
) -> api_types.FeedStatus:
    """Derive one feed's status from its manifest entry with the feed rules.

    The rules are those of `core.feeds.Feed`: the frame's age is measured from
    its fetch timestamp; it is expired when the manifest's next fetch time has
    passed, or, with no next fetch time, when the feed has a refresh cadence at
    all; it is healthy while its age is within the cadence plus
    `HEALTH_CHECK_BUFFER`, and always when it never refreshes. A carried-forward
    entry keeps its original fetch timestamp, so its age keeps growing and the
    health rule marks it exactly as the fetching feed would have.

    Args:
        feed_name: The feed the entry describes.
        location_code: The location's code, reported on the status.
        feed: The manifest entry.
        frame: The served frame, summarized on the status.
        now: The naive UTC instant to measure age and expiry against.

    Returns:
        The feed's status. `name` is the feed name's value: the manifest
        carries no feed class name.
    """
    fetch_timestamp = _naive_utc(feed.fetch_timestamp)
    next_fetch_after = (
        None if feed.next_fetch_after is None else _naive_utc(feed.next_fetch_after)
    )
    age = now - fetch_timestamp
    if next_fetch_after is not None:
        is_expired = now > next_fetch_after
        seconds_until_next_fetch = max(0.0, (next_fetch_after - now).total_seconds())
    else:
        is_expired = feed.expiration_seconds is not None
        seconds_until_next_fetch = None
    if feed.expiration_seconds is None:
        is_healthy = True
    else:
        is_healthy = age <= (
            datetime.timedelta(seconds=feed.expiration_seconds) + HEALTH_CHECK_BUFFER
        )
    return api_types.FeedStatus(
        name=feed_name.value,
        location=location_code,
        fetch_timestamp=fetch_timestamp,
        next_fetch_after=next_fetch_after,
        age_seconds=age.total_seconds(),
        seconds_until_next_fetch=seconds_until_next_fetch,
        consecutive_failures=feed.consecutive_failures,
        is_expired=is_expired,
        is_healthy=is_healthy,
        expiration_seconds=feed.expiration_seconds,
        data_summary=summarize_dataframe(frame),
        error=feed.last_error,
        historical_temp_status=feed.historical,
    )


class SnapshotLocationManager:
    """One location's serving state from a loaded generation; see the module."""

    def __init__(
        self,
        config: config_lib.LocationConfig,
        location: LocationManifest,
        frames: dict[FeedName, pd.DataFrame],
        plots: dict[PlotName, bytes],
        loaded_at: datetime.datetime,
    ) -> None:
        """Build the serving state for one location.

        Args:
            config: The location's configuration.
            location: The location's manifest entry from the loaded generation.
            frames: The location's restored frames, keyed like the manifest.
            plots: The location's plot bytes, keyed like the manifest.
            loaded_at: When the generation was loaded, timezone-aware.

        Raises:
            ValueError: If a manifest feed has no frame or a manifest plot has
                no bytes, which the loader guarantees against.
        """
        if loaded_at.tzinfo is None:
            raise ValueError("loaded_at must be timezone-aware")
        missing_frames = set(location.feeds) - set(frames)
        if missing_frames:
            raise ValueError(f"Manifest feeds without frames: {sorted(missing_frames)}")
        missing_plots = set(location.plots) - set(plots)
        if missing_plots:
            raise ValueError(f"Manifest plots without bytes: {sorted(missing_plots)}")

        self.config = config
        self.loaded_at = loaded_at
        self._manifest = location
        self._feeds: dict[FeedName, FeedSnapshot] = {
            feed_name: FeedSnapshot(frame=frames[feed_name], metadata=feed_object)
            for feed_name, feed_object in location.feeds.items()
        }
        self._plots: dict[PlotName, bytes] = {
            plot_name: plots[plot_name] for plot_name in location.plots
        }
        self._tide_prediction_frame = self._build_tide_prediction_frame()
        self._current_prediction_frame = self._build_current_prediction_frame()

    def _build_tide_prediction_frame(self) -> pd.DataFrame | None:
        """Derive the tide-height curve once, as the fetching manager does."""
        if self.config.tide_source is None:
            return None
        tides = self._feeds.get(FeedName.TIDES)
        if tides is None:
            return None
        try:
            return queries.prepare_tide_prediction_frame(tides.values)
        except queries.DataUnavailableError as e:
            logging.warning(
                f"[{self.config.code}] Tide prediction frame unavailable: {e}"
            )
            return None

    def _build_current_prediction_frame(self) -> pd.DataFrame | None:
        """Derive the current prediction columns once for prediction sources."""
        currents_source = self.config.currents_source
        if (
            currents_source is None
            or currents_source.source_type != DataSourceType.PREDICTION
        ):
            return None
        currents = self._feeds.get(FeedName.CURRENTS)
        if currents is None:
            return None
        return queries.prepare_current_prediction_frame(currents.values)

    @property
    def has_data(self) -> bool:
        """Whether any feed of this location holds data."""
        return bool(self._feeds)

    def has_feed(self, feed_name: FeedName | str) -> bool:
        """Whether `feed_name` names a feed, configured for this location or not."""
        try:
            FeedName(feed_name)
        except ValueError:
            return False
        return True

    def has_feed_data(self, feed_name: FeedName) -> bool:
        """Whether the generation holds a frame for the feed."""
        return feed_name in self._feeds

    def get_feed_values(self, feed_name: FeedName | str) -> pd.DataFrame:
        """Return the feed's served frame.

        Raises:
            KeyError: If the feed name is unknown.
            DataUnavailableError: If the generation holds no frame for it.
        """
        try:
            normalized = FeedName(feed_name)
        except ValueError as e:
            raise KeyError(feed_name) from e
        feed = self._feeds.get(normalized)
        if feed is None:
            raise queries.DataUnavailableError(
                f"Feed '{normalized}' data not available"
            )
        return feed.values

    def get_plot(self, plot_type: PlotName) -> bytes | None:
        """Return the plot's SVG bytes, or None if the generation has none."""
        return self._plots.get(plot_type)

    @property
    def status(self) -> api_types.LocationStatus:
        """Per-feed status derived from the manifest at the current instant."""
        now = utc_now()
        return api_types.LocationStatus(
            feeds={
                feed_name.value: feed_status_from_manifest(
                    feed_name, self.config.code, feed.metadata, feed.frame, now
                )
                for feed_name, feed in self._feeds.items()
            }
        )

    def get_current_temperature(self) -> TemperatureReading:
        """See `queries.get_current_temperature`."""
        return queries.get_current_temperature(self._feeds)

    def get_tide_info_at_time(self, t: datetime.datetime | None = None) -> TideInfo:
        """See `queries.get_tide_info_at_time`."""
        return queries.get_tide_info_at_time(self._feeds, self.config, t)

    def predict_tide_at_time(
        self, t: datetime.datetime | None = None
    ) -> TideState | None:
        """Estimate tide state from the derived curve, or None without one."""
        if self._tide_prediction_frame is None:
            return None
        return queries.predict_tide_from_precomputed_frame(
            self._tide_prediction_frame, self.config, t
        )

    def get_chart_info(self, t: datetime.datetime | None = None) -> LegacyChartInfo:
        """See `queries.get_chart_info`."""
        return queries.get_chart_info(self._feeds, self.config, t)

    def get_current_flow_info(self) -> CurrentInfo:
        """See `queries.get_current_flow_info`."""
        return queries.get_current_flow_info(self._feeds)

    def predict_flow_at_time(self, t: datetime.datetime | None = None) -> CurrentInfo:
        """Predict current from the derived frame, else from the raw frame."""
        if self._current_prediction_frame is not None:
            return queries.predict_flow_from_precomputed_frame(
                self._current_prediction_frame, self.config, t
            )
        return queries.predict_flow_at_time(self._feeds, self.config, t)
