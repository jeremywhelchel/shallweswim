mock_provider "google" {}

run "monitoring_plan" {
  command = plan

  variables {
    project_id   = "test-project"
    service_name = "shallweswim"
  }

  assert {
    condition = alltrue([
      for metric in [
        google_logging_metric.feed_updates,
        google_logging_metric.feed_update_duration,
        google_logging_metric.feed_records,
        google_logging_metric.plot_generations,
        google_logging_metric.plot_availability_latency,
      ] : strcontains(metric.filter, "resource.labels.service_name=\"shallweswim\"")
    ])
    error_message = "Every metric must be scoped to the configured Cloud Run service."
  }

  assert {
    condition = (
      length(google_logging_metric.feed_updates.metric_descriptor[0].labels) == 4 &&
      length(google_logging_metric.plot_generations.metric_descriptor[0].labels) == 3
    )
    error_message = "Metric label sets must remain bounded by the reviewed contracts."
  }

  assert {
    condition = (
      google_logging_metric.feed_update_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.feed_records.value_extractor == "EXTRACT(jsonPayload.record_count)" &&
      google_logging_metric.plot_availability_latency.value_extractor == "EXTRACT(jsonPayload.duration_ms)"
    )
    error_message = "Distribution metrics must extract the reviewed numeric JSON fields."
  }

  assert {
    condition     = jsondecode(google_monitoring_dashboard.operations.dashboard_json).displayName == "Shall We Swim Operations [Terraform]"
    error_message = "The managed dashboard must retain its visible Terraform ownership marker."
  }
}
