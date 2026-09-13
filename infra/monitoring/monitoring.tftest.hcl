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
        google_logging_metric.archive_merges,
        google_logging_metric.archive_merge_duration,
        google_logging_metric.archive_merge_new_rows,
        google_logging_metric.archive_merge_revised_rows,
        google_logging_metric.updater_runs,
        google_logging_metric.updater_run_duration,
        google_logging_metric.snapshot_publishes,
        google_logging_metric.snapshot_publish_duration,
      ] : strcontains(metric.filter, "resource.labels.service_name=\"shallweswim\"")
    ])
    error_message = "Every metric must be scoped to the configured Cloud Run service."
  }

  assert {
    condition = alltrue([
      for metric in [
        google_logging_metric.feed_updates,
        google_logging_metric.feed_update_duration,
        google_logging_metric.feed_records,
        google_logging_metric.plot_generations,
        google_logging_metric.plot_availability_latency,
        google_logging_metric.archive_merges,
        google_logging_metric.archive_merge_duration,
        google_logging_metric.archive_merge_new_rows,
        google_logging_metric.archive_merge_revised_rows,
        google_logging_metric.updater_runs,
        google_logging_metric.updater_run_duration,
        google_logging_metric.snapshot_publishes,
        google_logging_metric.snapshot_publish_duration,
      ] : strcontains(metric.filter, "resource.labels.job_name=\"shallweswim-capture\"")
    ])
    error_message = "Every metric must also match the capture job, which is the archive's only production writer."
  }

  assert {
    condition = (
      length(google_logging_metric.feed_updates.metric_descriptor[0].labels) == 4 &&
      length(google_logging_metric.plot_generations.metric_descriptor[0].labels) == 3 &&
      length(google_logging_metric.archive_merges.metric_descriptor[0].labels) == 2 &&
      length(google_logging_metric.archive_merge_duration.metric_descriptor[0].labels) == 2 &&
      length(google_logging_metric.archive_merge_new_rows.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.archive_merge_revised_rows.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.updater_runs.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.updater_run_duration.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_publishes.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_publish_duration.metric_descriptor[0].labels) == 1
    )
    error_message = "Metric label sets must remain bounded by the reviewed contracts."
  }

  assert {
    condition = (
      google_logging_metric.feed_update_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.feed_records.value_extractor == "EXTRACT(jsonPayload.record_count)" &&
      google_logging_metric.plot_availability_latency.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.archive_merge_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.updater_run_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.snapshot_publish_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.archive_merge_new_rows.value_extractor == "EXTRACT(jsonPayload.new_count)" &&
      google_logging_metric.archive_merge_revised_rows.value_extractor == "EXTRACT(jsonPayload.revised_count)"
    )
    error_message = "Distribution metrics must extract the reviewed numeric JSON fields."
  }

  assert {
    condition = alltrue([
      for metric in [
        google_logging_metric.updater_runs,
        google_logging_metric.snapshot_publishes,
        google_logging_metric.archive_merge_new_rows,
        google_logging_metric.archive_merge_revised_rows,
        google_logging_metric.archive_merge_duration,
        ] : strcontains(
        google_monitoring_dashboard.operations.dashboard_json,
        "${local.metric_prefix}/${metric.name}"
      )
    ])
    error_message = "The operations dashboard must show the capture run, snapshot publish, and archive row metrics."
  }

  assert {
    condition = (
      google_monitoring_alert_policy.capture_job_heartbeat.conditions[0].condition_absent[0].duration == "10800s" &&
      strcontains(
        google_monitoring_alert_policy.capture_job_heartbeat.conditions[0].condition_absent[0].filter,
        "resource.type = \"cloud_run_job\""
      ) &&
      strcontains(
        google_monitoring_alert_policy.archive_merge_failures.conditions[0].condition_threshold[0].filter,
        "resource.type = \"cloud_run_job\""
      )
    )
    error_message = "The capture job policies must watch the job resource with the reviewed 3 hour heartbeat window."
  }

  assert {
    condition     = jsondecode(google_monitoring_dashboard.operations.dashboard_json).displayName == "Shall We Swim Operations [Terraform]"
    error_message = "The managed dashboard must retain its visible Terraform ownership marker."
  }

  assert {
    condition = strcontains(
      google_monitoring_dashboard.operations.dashboard_json,
      "${local.metric_prefix}/${google_logging_metric.archive_merges.name}"
    )
    error_message = "The operations dashboard must show the archive merge metric."
  }

  assert {
    condition = alltrue([
      for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
      !strcontains(try(tile.widget.xyChart.dataSets[0].timeSeriesQuery.timeSeriesFilter.filter, ""), "resource.type=")
      if strcontains(
        try(tile.widget.xyChart.dataSets[0].timeSeriesQuery.timeSeriesFilter.filter, ""),
        google_logging_metric.archive_merges.name
      )
    ])
    error_message = "The archive merge chart must not pin resource.type, or capture job series are hidden."
  }

  assert {
    condition = (
      length([
        for query in flatten([
          for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
          [for data_set in tile.widget.xyChart.dataSets : try(data_set.timeSeriesQuery.timeSeriesQueryLanguage, "")]
        ]) : query if query != ""
      ]) == 2 &&
      alltrue([
        for query in flatten([
          for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
          [for data_set in tile.widget.xyChart.dataSets : try(data_set.timeSeriesQuery.timeSeriesQueryLanguage, "")]
        ]) : strcontains(query, "sum_from(") if query != ""
      ])
    )
    error_message = "The two distribution-valued observation counters must be charted with MQL sum_from, which timeSeriesFilter cannot express."
  }

  assert {
    condition = (
      length(jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles) == 11 &&
      alltrue([
        for title in [
          "Archive merges per hour by outcome",
          "Snapshot publishes per hour by outcome",
          "New observations per hour by source (estimated)",
          "Revised observations per hour by source (estimated)",
          ] : contains([
            for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
            tile.widget.title
        ], title)
      ]) &&
      !anytrue([
        for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
        strcontains(tile.widget.title, "Failed archive merges")
      ])
    )
    error_message = "The dashboard must keep eleven tiles, fold failed merges into the hourly outcome stack, and mark the estimated observation counts."
  }

  assert {
    condition = alltrue([
      for policy in [
        google_monitoring_alert_policy.live_feed_update_latency,
        google_monitoring_alert_policy.live_plot_availability_latency,
        google_monitoring_alert_policy.repeated_feed_failures,
        google_monitoring_alert_policy.plot_generation_failure,
        google_monitoring_alert_policy.capture_job_heartbeat,
        google_monitoring_alert_policy.archive_merge_failures,
        ] : (
        startswith(policy.display_name, "[Terraform][Shadow]") &&
        length(policy.notification_channels) == 0 &&
        policy.user_labels.mode == "shadow"
      )
    ])
    error_message = "Baseline alert policies must remain visibly marked as shadow policies without notification channels."
  }
}
