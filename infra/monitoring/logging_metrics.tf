locals {
  # Both the web service and the capture job emit the structured events these
  # metrics extract. Newline-separated terms are implicit AND in the Cloud
  # Logging filter grammar, so the service/job alternation stays inside one
  # parenthesized OR group that the appended jsonPayload terms AND against.
  application_log_filter = <<-EOT
    ((resource.type="cloud_run_revision" AND resource.labels.service_name="${var.service_name}") OR (resource.type="cloud_run_job" AND resource.labels.job_name="${var.job_name}"))
  EOT

  feed_labels = {
    location = "EXTRACT(jsonPayload.location)"
    feed     = "EXTRACT(jsonPayload.feed)"
    provider = "EXTRACT(jsonPayload.provider)"
    outcome  = "EXTRACT(jsonPayload.outcome)"
  }

  plot_labels = {
    location = "EXTRACT(jsonPayload.location)"
    feed     = "EXTRACT(jsonPayload.feed)"
    outcome  = "EXTRACT(jsonPayload.outcome)"
  }

  archive_labels = {
    source  = "EXTRACT(jsonPayload.source_identity)"
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  freshness_labels = {
    location = "EXTRACT(jsonPayload.location)"
    feed     = "EXTRACT(jsonPayload.feed)"
    outcome  = "EXTRACT(jsonPayload.outcome)"
  }

  # One completion event per partition merge, one summary event per capture run,
  # one event per snapshot publish attempt, one freshness event per location
  # and configured feed per publish, and one event per bundle load attempt by
  # the web service. Several metrics extract different numbers from each of
  # them. A feed with nothing to serve reports outcome=absent and no age, so
  # it contributes no freshness sample.
  archive_merge_filter      = "${local.application_log_filter}\njsonPayload.component=\"archive\"\njsonPayload.operation=\"merge\""
  updater_run_filter        = "${local.application_log_filter}\njsonPayload.component=\"updater\"\njsonPayload.operation=\"run\""
  snapshot_publish_filter   = "${local.application_log_filter}\njsonPayload.component=\"snapshot\"\njsonPayload.operation=\"publish\""
  snapshot_freshness_filter = "${local.application_log_filter}\njsonPayload.component=\"snapshot\"\njsonPayload.operation=\"freshness\""
  snapshot_load_filter      = "${local.application_log_filter}\njsonPayload.component=\"snapshot\"\njsonPayload.operation=\"load\""
}

resource "google_logging_metric" "feed_updates" {
  name        = "shallweswim_feed_updates"
  description = "Completed feed update attempts by bounded outcome. Managed by Terraform."
  filter      = "${local.application_log_filter}\njsonPayload.operation=\"feed_update\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "location"
      value_type  = "STRING"
      description = "Configured swimming location code."
    }
    labels {
      key         = "feed"
      value_type  = "STRING"
      description = "Semantic feed name."
    }
    labels {
      key         = "provider"
      value_type  = "STRING"
      description = "Bounded upstream provider family."
    }
    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, unavailable, or failed."
    }
  }

  label_extractors = local.feed_labels
}

resource "google_logging_metric" "feed_update_duration" {
  name            = "shallweswim_feed_update_duration_ms"
  description     = "Feed update attempt duration in milliseconds. Managed by Terraform."
  filter          = "${local.application_log_filter}\njsonPayload.operation=\"feed_update\""
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    dynamic "labels" {
      for_each = toset(["location", "feed", "provider", "outcome"])
      content {
        key        = labels.value
        value_type = "STRING"
      }
    }
  }

  label_extractors = local.feed_labels

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "feed_records" {
  name            = "shallweswim_feed_records"
  description     = "Records published by successful feed updates. Managed by Terraform."
  filter          = "${local.application_log_filter}\njsonPayload.operation=\"feed_update\"\njsonPayload.outcome=\"success\""
  value_extractor = "EXTRACT(jsonPayload.record_count)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "{record}"

    dynamic "labels" {
      for_each = toset(["location", "feed", "provider"])
      content {
        key        = labels.value
        value_type = "STRING"
      }
    }
  }

  label_extractors = {
    location = local.feed_labels.location
    feed     = local.feed_labels.feed
    provider = local.feed_labels.provider
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 12
      growth_factor      = 4
      scale              = 1
    }
  }
}

resource "google_logging_metric" "plot_generations" {
  name        = "shallweswim_plot_generations"
  description = "Completed background plot generations by bounded outcome. Managed by Terraform."
  filter      = "${local.application_log_filter}\njsonPayload.operation=\"plot_generation\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    dynamic "labels" {
      for_each = toset(["location", "feed", "outcome"])
      content {
        key        = labels.value
        value_type = "STRING"
      }
    }
  }

  label_extractors = local.plot_labels
}

resource "google_logging_metric" "plot_availability_latency" {
  name            = "shallweswim_plot_availability_latency_ms"
  description     = "Plot submit-to-harvest availability latency in milliseconds. Managed by Terraform."
  filter          = "${local.application_log_filter}\njsonPayload.operation=\"plot_generation\""
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    dynamic "labels" {
      for_each = toset(["location", "feed", "outcome"])
      content {
        key        = labels.value
        value_type = "STRING"
      }
    }
  }

  label_extractors = local.plot_labels

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "archive_merges" {
  name        = "shallweswim_archive_merges"
  description = "Completed archive partition merges by bounded outcome. Managed by Terraform."
  filter      = local.archive_merge_filter

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "source"
      value_type  = "STRING"
      description = "Permanent archive source identity."
    }
    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, unchanged, or failed."
    }
  }

  label_extractors = local.archive_labels
}

resource "google_logging_metric" "updater_runs" {
  name        = "shallweswim_updater_runs"
  description = "Completed capture runs by bounded outcome. Managed by Terraform."
  filter      = local.updater_run_filter

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, partial, or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }
}

resource "google_logging_metric" "updater_run_duration" {
  name            = "shallweswim_updater_run_duration_ms"
  description     = "Capture run duration in milliseconds. Managed by Terraform."
  filter          = local.updater_run_filter
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, partial, or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "archive_merge_duration" {
  name            = "shallweswim_archive_merge_duration_ms"
  description     = "Archive partition merge duration in milliseconds. Managed by Terraform."
  filter          = local.archive_merge_filter
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    dynamic "labels" {
      for_each = toset(["source", "outcome"])
      content {
        key        = labels.value
        value_type = "STRING"
      }
    }
  }

  label_extractors = local.archive_labels

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "archive_merge_new_rows" {
  name            = "shallweswim_archive_merge_new_rows"
  description     = "Observations added to the archive per merge. Managed by Terraform."
  filter          = local.archive_merge_filter
  value_extractor = "EXTRACT(jsonPayload.new_count)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "{record}"

    labels {
      key         = "source"
      value_type  = "STRING"
      description = "Permanent archive source identity."
    }
  }

  label_extractors = {
    source = local.archive_labels.source
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 12
      growth_factor      = 4
      scale              = 1
    }
  }
}

resource "google_logging_metric" "archive_merge_revised_rows" {
  name            = "shallweswim_archive_merge_revised_rows"
  description     = "Upstream corrections applied to the archive per merge. Managed by Terraform."
  filter          = local.archive_merge_filter
  value_extractor = "EXTRACT(jsonPayload.revised_count)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "{record}"

    labels {
      key         = "source"
      value_type  = "STRING"
      description = "Permanent archive source identity."
    }
  }

  label_extractors = {
    source = local.archive_labels.source
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 12
      growth_factor      = 4
      scale              = 1
    }
  }
}

resource "google_logging_metric" "snapshot_publishes" {
  name        = "shallweswim_snapshot_publishes"
  description = "Completed snapshot publish attempts by bounded outcome. Managed by Terraform."
  filter      = local.snapshot_publish_filter

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, unchanged, or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }
}

resource "google_logging_metric" "snapshot_publish_duration" {
  name            = "shallweswim_snapshot_publish_duration_ms"
  description     = "Snapshot publish attempt duration in milliseconds. Managed by Terraform."
  filter          = local.snapshot_publish_filter
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success, unchanged, or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "snapshot_feed_age" {
  name            = "shallweswim_snapshot_feed_age_seconds"
  description     = "Age of the feed frame each published generation serves, in seconds. Managed by Terraform."
  filter          = local.snapshot_freshness_filter
  value_extractor = "EXTRACT(jsonPayload.age_seconds)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "s"

    labels {
      key         = "location"
      value_type  = "STRING"
      description = "Configured swimming location code."
    }
    labels {
      key         = "feed"
      value_type  = "STRING"
      description = "Semantic feed name."
    }
    labels {
      key        = "outcome"
      value_type = "STRING"
      # Label descriptions are immutable on a log-based metric: changing this
      # string replaces the metric and erases its history. The outcome set is
      # documented in README.md; it now includes held.
      description = "One of success or carried; absent carries no age."
    }
  }

  label_extractors = local.freshness_labels

  # Twenty doubling buckets from one second reach twelve days, past every
  # per-feed threshold a carried-forward feed crosses.
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "snapshot_loads" {
  name        = "shallweswim_snapshot_loads"
  description = "Completed bundle load attempts by bounded outcome. Managed by Terraform."
  filter      = local.snapshot_load_filter

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }
}

resource "google_logging_metric" "snapshot_load_duration" {
  name            = "shallweswim_snapshot_load_duration_ms"
  description     = "Bundle load attempt duration in milliseconds. Managed by Terraform."
  filter          = local.snapshot_load_filter
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}

resource "google_logging_metric" "snapshot_load_lag" {
  name            = "shallweswim_snapshot_load_lag_seconds"
  description     = "Age of the loaded generation's publication time at load, in seconds. Managed by Terraform."
  filter          = local.snapshot_load_filter
  value_extractor = "EXTRACT(jsonPayload.age_seconds)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "s"

    labels {
      key         = "outcome"
      value_type  = "STRING"
      description = "One of success or failed."
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  # Same exponential buckets as snapshot_feed_age: twenty doublings from one
  # second reach well past any plausible load lag.
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}
