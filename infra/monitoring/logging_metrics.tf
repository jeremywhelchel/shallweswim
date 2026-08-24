locals {
  application_log_filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${var.service_name}"
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
