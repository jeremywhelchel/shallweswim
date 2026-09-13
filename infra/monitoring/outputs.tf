output "dashboard_id" {
  description = "Terraform-managed operations dashboard resource ID."
  value       = google_monitoring_dashboard.operations.id
}

output "logging_metric_names" {
  description = "User-defined log-based metrics created by this module."
  value = [
    google_logging_metric.feed_updates.name,
    google_logging_metric.feed_update_duration.name,
    google_logging_metric.feed_records.name,
    google_logging_metric.plot_generations.name,
    google_logging_metric.plot_availability_latency.name,
    google_logging_metric.archive_merges.name,
    google_logging_metric.archive_merge_duration.name,
    google_logging_metric.archive_merge_new_rows.name,
    google_logging_metric.archive_merge_revised_rows.name,
    google_logging_metric.updater_runs.name,
    google_logging_metric.updater_run_duration.name,
  ]
}
