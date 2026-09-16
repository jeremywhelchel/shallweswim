locals {
  # The web servers load and serve generations, so the snapshot load policies
  # watch the service revisions. Everything else the job does: it is the only
  # process that fetches feeds, draws plots on a schedule, and writes the
  # archive, so the feed, plot, archive, and run policies watch the job.
  cloud_run_resource_filter     = "resource.type = \"cloud_run_revision\""
  cloud_run_job_resource_filter = "resource.type = \"cloud_run_job\""
  paging_alert_labels = {
    managed_by = "terraform"
    mode       = "paging"
  }

  # Per-feed-type snapshot freshness thresholds, in seconds, from the existing
  # feed health rule: the feed's expiration interval plus 15 minutes. A feed
  # whose published frame is older than its threshold is being carried forward
  # across runs rather than refreshed. The capture job publishes every ten
  # minutes, so an hour of samples holds six values per feed and location.
  snapshot_freshness_thresholds = {
    live_temps     = 1500
    historic_temps = 11700
    tides          = 87300
    currents       = 87300
  }
}

resource "google_monitoring_alert_policy" "live_feed_update_latency" {
  display_name          = "[Terraform] Live feed update latency"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "A live temperature fetch took longer than 45 seconds. The matched log entry names the location and the exact duration; a single slow provider answer is expected now and then, a run of them is a provider or network problem."
  }

  # Matches the events rather than the duration distribution, whose doubling
  # buckets round a percentile up to the next bucket edge (a 33-second fetch
  # reads as 65 seconds); the event carries the exact duration_ms.
  conditions {
    display_name = "Live feed update over 45s"

    condition_matched_log {
      filter = "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${var.job_name}\" AND jsonPayload.operation=\"feed_update\" AND jsonPayload.feed=\"live_temps\" AND jsonPayload.duration_ms > 45000"

      label_extractors = {
        location = "EXTRACT(jsonPayload.location)"
      }
    }
  }

  # A log-matching condition notifies per matching entry, rate-limited to one
  # notification an hour, and the incident closes on its own once no entry
  # has matched for a while.
  alert_strategy {
    auto_close = "1800s"

    notification_rate_limit {
      period = "3600s"
    }
  }
}

resource "google_monitoring_alert_policy" "live_plot_availability_latency" {
  display_name          = "[Terraform] Live plot availability latency"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "A live temperature plot took longer than 45 seconds from submission to harvest. The harvest waits for the location's whole cycle, so one slow fetch stamps every plot of that run with its delay; check the feed update durations of the same run first."
  }

  # Matches the events rather than the duration distribution, whose doubling
  # buckets round a percentile up to the next bucket edge (a 33-second fetch
  # reads as 65 seconds); the event carries the exact duration_ms.
  conditions {
    display_name = "Live plot availability over 45s"

    condition_matched_log {
      filter = "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${var.job_name}\" AND jsonPayload.operation=\"plot_generation\" AND jsonPayload.feed=\"live_temps\" AND jsonPayload.duration_ms > 45000"

      label_extractors = {
        location = "EXTRACT(jsonPayload.location)"
      }
    }
  }

  # A log-matching condition notifies per matching entry, rate-limited to one
  # notification an hour, and the incident closes on its own once no entry
  # has matched for a while.
  alert_strategy {
    auto_close = "1800s"

    notification_rate_limit {
      period = "3600s"
    }
  }
}

resource "google_monitoring_alert_policy" "repeated_feed_failures" {
  display_name          = "[Terraform] Repeated feed failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "ERROR"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "at least three unexpected failures for the same feed and location in 10 minutes. Expected station-unavailable outcomes are excluded."
  }

  conditions {
    display_name = "At least 3 feed failures in 10m"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.feed_updates.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.outcome = \"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 2
      duration        = "0s"

      aggregations {
        alignment_period     = "600s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.location", "metric.label.feed"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "plot_generation_failure" {
  display_name          = "[Terraform] Plot generation failure"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "ERROR"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "a plot generation completed with a failed outcome."
  }

  conditions {
    display_name = "Any plot generation failure"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.plot_generations.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.outcome = \"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.location", "metric.label.feed"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "snapshot_load_lag" {
  display_name          = "[Terraform] Snapshot load lag"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "A web server loaded a generation more than 1800 seconds after it was published, three job cadences; the ordinary lag is under a minute and a deploy's first load is about ten minutes. This catches an instance whose refresh path is stuck, not a job that stopped publishing, which the capture job heartbeat catches."
  }

  # Matches the load events rather than the lag distribution, whose doubling
  # buckets round a percentile up to the next bucket edge (an 1,100-second
  # lag reads as 2,048); the event carries the exact age_seconds.
  conditions {
    display_name = "Snapshot load lag over 1800s"

    condition_matched_log {
      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND jsonPayload.operation=\"load\" AND jsonPayload.age_seconds > 1800"
    }
  }

  # A log-matching condition notifies per matching entry, rate-limited to one
  # notification an hour, and the incident closes on its own once no entry
  # has matched for a while.
  alert_strategy {
    auto_close = "1800s"

    notification_rate_limit {
      period = "3600s"
    }
  }
}

resource "google_monitoring_alert_policy" "snapshot_load_failures" {
  display_name          = "[Terraform] Snapshot load failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "the web service logged three or more failed bundle loads within fifteen minutes. A single failure retries a check interval later and the instance keeps serving its loaded generation; repeated failures mean an instance cannot read the store at all. The load lag policy catches slowness, this catches inability."
  }

  conditions {
    display_name = "At least 3 failed snapshot loads in 15m"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.snapshot_loads.name}\" AND ${local.cloud_run_resource_filter} AND metric.label.outcome = \"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 2
      duration        = "0s"

      aggregations {
        alignment_period     = "900s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "capture_job_heartbeat" {
  display_name          = "[Terraform] Capture job heartbeat"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "CRITICAL"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "The ten-minute capture job reported no success or partial run summary for 30 minutes, which is three missed runs. A partial run still proves the job executed. The web servers serve only what this job publishes, so the site is now aging silently: check the job's executions in Cloud Run and its run summaries in the logs (MONITORING.md). The condition evaluates only once the run counter has data."
  }

  conditions {
    display_name = "No successful or partial capture run in 30m"

    condition_absent {
      filter   = "metric.type = \"${local.metric_prefix}/${google_logging_metric.updater_runs.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.outcome = one_of(\"success\", \"partial\")"
      duration = "1800s"

      aggregations {
        alignment_period     = "600s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "archive_merge_failures" {
  display_name          = "[Terraform] Archive merge failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "an archive partition merge completed with a failed outcome in the last hour. A conflicting equally recent claim is self-recovering; repeated failures mean the store or the merge path is broken."
  }

  conditions {
    display_name = "Any failed archive merge in 1h"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.archive_merges.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.outcome = \"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period     = "3600s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.source"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "snapshot_feed_freshness" {
  for_each = local.snapshot_freshness_thresholds

  display_name          = "[Terraform] Snapshot ${each.key} freshness"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "WARNING"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "The published snapshot served a `${each.key}` frame older than ${each.value} seconds, its expiration interval plus 15 minutes, which means the feed is being carried forward instead of refreshed. The matched log entry names the location and the exact age."
  }

  # This condition matches the freshness events themselves rather than the
  # snapshot_feed_age_seconds distribution, because the distribution's
  # buckets double in size and a percentile of it rounds an age up to the
  # next bucket edge: a normal 10,800-second historical hold reads as about
  # 16,000 seconds and would trip an 11,700-second threshold on every hold.
  # The event carries the exact age_seconds, so the comparison is exact.
  conditions {
    display_name = "Snapshot ${each.key} age > ${each.value}s"

    condition_matched_log {
      filter = "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${var.job_name}\" AND jsonPayload.component=\"snapshot\" AND jsonPayload.operation=\"freshness\" AND jsonPayload.feed=\"${each.key}\" AND jsonPayload.age_seconds > ${each.value}"

      label_extractors = {
        location = "EXTRACT(jsonPayload.location)"
        feed     = "EXTRACT(jsonPayload.feed)"
      }
    }
  }

  # A log-matching condition notifies per matching entry, so one stale feed
  # publishing every ten minutes is rate-limited to one notification an hour
  # and the incident closes on its own once no entry has matched for a while.
  alert_strategy {
    auto_close = "1800s"

    notification_rate_limit {
      period = "3600s"
    }
  }
}
