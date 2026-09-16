locals {
  # The web servers load and serve generations, so the snapshot load policies
  # watch the service revisions. Everything else the job does: it is the only
  # process that fetches feeds, draws plots on a schedule, and writes the
  # archive, so the feed, plot, archive, and run policies watch the job.
  cloud_run_resource_filter     = "resource.type = \"cloud_run_revision\""
  cloud_run_job_resource_filter = "resource.type = \"cloud_run_job\""
  shadow_alert_labels = {
    managed_by = "terraform"
    mode       = "shadow"
  }
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
  display_name          = "[Terraform][Shadow] Live feed update latency"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: live feed-update p95 exceeded 45 seconds for 10 minutes. This policy deliberately sends no notifications while its threshold is baselined."
  }

  conditions {
    display_name = "Live feed-update p95 > 45s for 10m"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.feed_update_duration.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.feed = \"live_temps\""
      comparison      = "COMPARISON_GT"
      threshold_value = 45000
      duration        = "600s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["metric.label.location", "metric.label.feed"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "live_plot_availability_latency" {
  display_name          = "[Terraform][Shadow] Live plot availability latency"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: live plot submit-to-harvest p95 exceeded 45 seconds for 10 minutes. This policy deliberately sends no notifications while its threshold is baselined."
  }

  conditions {
    display_name = "Live plot availability p95 > 45s for 10m"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.plot_availability_latency.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.feed = \"live_temps\""
      comparison      = "COMPARISON_GT"
      threshold_value = 45000
      duration        = "600s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["metric.label.location", "metric.label.feed"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "repeated_feed_failures" {
  display_name          = "[Terraform][Shadow] Repeated feed failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "ERROR"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: at least three unexpected failures for the same feed and location in 10 minutes. Expected station-unavailable outcomes are excluded."
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
  display_name          = "[Terraform][Shadow] Plot generation failure"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "ERROR"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: a plot generation completed with a failed outcome. This policy deliberately sends no notifications during baseline evaluation."
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
  display_name          = "[Terraform][Shadow] Snapshot load lag"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: the web service loaded a bundle generation whose publication lag exceeded 1800 seconds in the last hour, three ten-minute capture job cadences; the observed lag is under a minute. This catches an instance whose refresh path is stuck, not a job that stopped publishing, which the capture job heartbeat catches. This policy deliberately sends no notifications while its threshold is baselined."
  }

  conditions {
    display_name = "Snapshot load lag p99 > 1800s in 1h"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.snapshot_load_lag.name}\" AND ${local.cloud_run_resource_filter}"
      comparison      = "COMPARISON_GT"
      threshold_value = 1800
      duration        = "0s"

      # The metric is a distribution, which has no max aligner; the hour's
      # 99th percentile approximates that hour's maximum load lag.
      aggregations {
        alignment_period     = "3600s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["metric.label.outcome"]
      }

      trigger {
        count = 1
      }
    }
  }
}

resource "google_monitoring_alert_policy" "snapshot_load_failures" {
  display_name          = "[Terraform][Shadow] Snapshot load failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: the web service logged three or more failed bundle loads within fifteen minutes. A single failure retries a check interval later and the instance keeps serving its loaded generation; repeated failures mean an instance cannot read the store at all. The load lag policy catches slowness, this catches inability. This policy deliberately sends no notifications while its threshold is baselined."
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
  display_name          = "[Terraform][Shadow] Archive merge failures"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: an archive partition merge completed with a failed outcome in the last hour. A conflicting equally recent claim is self-recovering, so this is a warn candidate rather than a page candidate. This policy deliberately sends no notifications during baseline evaluation."
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

  display_name          = "[Terraform][Shadow] Snapshot ${each.key} freshness"
  combiner              = "OR"
  enabled               = true
  notification_channels = []
  severity              = "WARNING"
  user_labels           = local.shadow_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "Shadow policy: the published snapshot served a `${each.key}` frame older than ${each.value} seconds, its expiration interval plus 15 minutes, which means the feed is being carried forward instead of refreshed. This policy deliberately sends no notifications while its threshold is baselined on real data."
  }

  conditions {
    display_name = "Snapshot ${each.key} age > ${each.value}s in 1h"

    condition_threshold {
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.snapshot_feed_age.name}\" AND ${local.cloud_run_job_resource_filter} AND metric.label.feed = \"${each.key}\""
      comparison      = "COMPARISON_GT"
      threshold_value = each.value
      duration        = "0s"

      # The metric is a distribution, which has no max aligner; with six
      # publishes per feed and location an hour, the hour's 99th percentile is
      # effectively that hour's maximum sample.
      aggregations {
        alignment_period     = "3600s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["metric.label.location", "metric.label.feed"]
      }

      trigger {
        count = 1
      }
    }
  }
}
