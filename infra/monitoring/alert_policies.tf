locals {
  cloud_run_resource_filter = "resource.type = \"cloud_run_revision\""
  shadow_alert_labels = {
    managed_by = "terraform"
    mode       = "shadow"
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
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.feed_update_duration.name}\" AND ${local.cloud_run_resource_filter} AND metric.label.feed = \"live_temps\""
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
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.plot_availability_latency.name}\" AND ${local.cloud_run_resource_filter} AND metric.label.feed = \"live_temps\""
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
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.feed_updates.name}\" AND ${local.cloud_run_resource_filter} AND metric.label.outcome = \"failed\""
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
      filter          = "metric.type = \"${local.metric_prefix}/${google_logging_metric.plot_generations.name}\" AND ${local.cloud_run_resource_filter} AND metric.label.outcome = \"failed\""
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
