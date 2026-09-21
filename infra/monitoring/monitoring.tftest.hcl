mock_provider "google" {}

run "monitoring_plan" {
  command = plan

  variables {
    project_id               = "test-project"
    service_name             = "shallweswim"
    notification_channel_ids = ["projects/test-project/notificationChannels/1"]
  }

  assert {
    condition = (
      google_logging_metric.slow_live_feed_updates.filter == "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"shallweswim-capture\" AND jsonPayload.operation=\"feed_update\" AND jsonPayload.feed=\"live_temps\" AND jsonPayload.duration_ms > 45000" &&
      google_logging_metric.slow_live_feed_updates.metric_descriptor[0].metric_kind == "DELTA" &&
      google_logging_metric.slow_live_feed_updates.metric_descriptor[0].value_type == "INT64" &&
      length(google_logging_metric.slow_live_feed_updates.metric_descriptor[0].labels) == 1 &&
      google_logging_metric.slow_live_feed_updates.label_extractors.location == "EXTRACT(jsonPayload.location)" &&
      strcontains(google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].filter, google_logging_metric.slow_live_feed_updates.name)
    )
    error_message = "Count exact slow live job events by location, including successful retries, for the latency policy."
  }

  assert {
    condition = (
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].threshold_value == 2 &&
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].comparison == "COMPARISON_GT" &&
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].duration == "0s" &&
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].aggregations[0].alignment_period == "1800s" &&
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].aggregations[0].per_series_aligner == "ALIGN_SUM" &&
      google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].aggregations[0].cross_series_reducer == "REDUCE_SUM" &&
      toset(google_monitoring_alert_policy.live_feed_update_latency.conditions[0].condition_threshold[0].aggregations[0].group_by_fields) == toset(["metric.label.location"])
    )
    error_message = "Slow updates must alert only at three per location within thirty minutes."
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
        google_logging_metric.snapshot_feed_age,
        google_logging_metric.snapshot_loads,
        google_logging_metric.snapshot_load_duration,
        google_logging_metric.snapshot_load_lag,
        google_logging_metric.snapshot_gcs,
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
        google_logging_metric.snapshot_feed_age,
        google_logging_metric.snapshot_loads,
        google_logging_metric.snapshot_load_duration,
        google_logging_metric.snapshot_load_lag,
        google_logging_metric.snapshot_gcs,
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
      length(google_logging_metric.snapshot_publish_duration.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_feed_age.metric_descriptor[0].labels) == 3 &&
      length(google_logging_metric.snapshot_loads.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_load_duration.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_load_lag.metric_descriptor[0].labels) == 1 &&
      length(google_logging_metric.snapshot_gcs.metric_descriptor[0].labels) == 1
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
      google_logging_metric.archive_merge_revised_rows.value_extractor == "EXTRACT(jsonPayload.revised_count)" &&
      google_logging_metric.snapshot_feed_age.value_extractor == "EXTRACT(jsonPayload.age_seconds)" &&
      google_logging_metric.snapshot_load_duration.value_extractor == "EXTRACT(jsonPayload.duration_ms)" &&
      google_logging_metric.snapshot_load_lag.value_extractor == "EXTRACT(jsonPayload.age_seconds)"
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
        google_logging_metric.snapshot_feed_age,
        google_logging_metric.snapshot_loads,
        google_logging_metric.snapshot_load_lag,
        google_logging_metric.snapshot_gcs,
        ] : strcontains(
        google_monitoring_dashboard.operations.dashboard_json,
        "${local.metric_prefix}/${metric.name}"
      )
    ])
    error_message = "The operations dashboard must show the capture run, snapshot publish, snapshot freshness, snapshot load, snapshot collection, and archive row metrics."
  }

  assert {
    condition = (
      google_monitoring_alert_policy.capture_job_heartbeat.conditions[0].condition_absent[0].duration == "1800s" &&
      strcontains(
        google_monitoring_alert_policy.capture_job_heartbeat.conditions[0].condition_absent[0].filter,
        "resource.type = \"cloud_run_job\""
      ) &&
      strcontains(
        google_monitoring_alert_policy.archive_merge_failures.conditions[0].condition_threshold[0].filter,
        "resource.type = \"cloud_run_job\""
      )
    )
    error_message = "The capture job policies must watch the job resource with the reviewed 30 minute heartbeat window."
  }

  assert {
    condition = alltrue(concat([
      for policy in [
        google_monitoring_alert_policy.live_feed_update_latency,
        google_monitoring_alert_policy.repeated_feed_failures,
        google_monitoring_alert_policy.plot_generation_failure,
      ] : strcontains(policy.conditions[0].condition_threshold[0].filter, "resource.type = \"cloud_run_job\"")
      ], [
      for policy in [
        google_monitoring_alert_policy.live_plot_availability_latency,
      ] : strcontains(policy.conditions[0].condition_matched_log[0].filter, "resource.type=\"cloud_run_job\"")
    ]))
    error_message = "The feed and plot policies must watch the job resource; only the job fetches feeds and draws scheduled plots."
  }

  assert {
    condition = alltrue([
      for tile in jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles :
      strcontains(try(tile.widget.xyChart.dataSets[0].timeSeriesQuery.timeSeriesFilter.filter, ""), "resource.type=\"cloud_run_job\"")
      if anytrue([
        for metric in [
          google_logging_metric.feed_updates.name,
          google_logging_metric.feed_update_duration.name,
          google_logging_metric.feed_records.name,
          google_logging_metric.plot_generations.name,
          google_logging_metric.plot_availability_latency.name,
        ] : strcontains(try(tile.widget.xyChart.dataSets[0].timeSeriesQuery.timeSeriesFilter.filter, ""), metric)
      ])
    ])
    error_message = "The feed and plot dashboard tiles must watch the job resource, which is the only emitter of those events."
  }

  assert {
    condition = (
      google_monitoring_alert_policy.snapshot_load_failures.conditions[0].condition_threshold[0].threshold_value == 2 &&
      google_monitoring_alert_policy.snapshot_load_failures.conditions[0].condition_threshold[0].aggregations[0].alignment_period == "900s" &&
      strcontains(
        google_monitoring_alert_policy.snapshot_load_failures.conditions[0].condition_threshold[0].filter,
        "metric.label.outcome = \"failed\""
      )
    )
    error_message = "The snapshot load failures policy must count failed loads over a fifteen minute window."
  }

  assert {
    condition = alltrue([
      strcontains(google_monitoring_alert_policy.snapshot_load_lag.conditions[0].condition_matched_log[0].filter, "jsonPayload.age_seconds > 1800"),
      strcontains(google_monitoring_alert_policy.snapshot_load_lag.conditions[0].condition_matched_log[0].filter, "resource.type=\"cloud_run_revision\""),
      strcontains(google_logging_metric.slow_live_feed_updates.filter, "jsonPayload.duration_ms > 45000"),
      strcontains(google_monitoring_alert_policy.live_plot_availability_latency.conditions[0].condition_matched_log[0].filter, "jsonPayload.duration_ms > 120000"),
    ])
    error_message = "The load lag and latency policies must match the exact event values against the reviewed thresholds."
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
      length(jsondecode(google_monitoring_dashboard.operations.dashboard_json).mosaicLayout.tiles) == 15 &&
      alltrue([
        for title in [
          "Archive merges per hour by outcome",
          "Snapshot publishes per hour by outcome",
          "Snapshot feed age max by feed per hour",
          "New observations per hour by source (estimated)",
          "Revised observations per hour by source (estimated)",
          "Snapshot loads per hour by outcome",
          "Snapshot load lag p99 per hour",
          "Snapshot collections per hour by outcome",
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
    error_message = "The dashboard must keep fifteen tiles, fold failed merges into the hourly outcome stack, and mark the estimated observation counts."
  }

  assert {
    condition = alltrue([
      for policy in [
        google_monitoring_alert_policy.live_feed_update_latency,
        google_monitoring_alert_policy.live_plot_availability_latency,
        google_monitoring_alert_policy.repeated_feed_failures,
        google_monitoring_alert_policy.plot_generation_failure,
        google_monitoring_alert_policy.snapshot_load_lag,
        google_monitoring_alert_policy.snapshot_load_failures,
        google_monitoring_alert_policy.capture_job_heartbeat,
        google_monitoring_alert_policy.archive_merge_failures,
        google_monitoring_alert_policy.application_errors,
        google_monitoring_alert_policy.request_5xx,
        google_monitoring_alert_policy.homepage_uptime_failure,
        google_monitoring_alert_policy.archive_read_gap,
        google_monitoring_alert_policy.snapshot_feed_stale,
        ] : (
        startswith(policy.display_name, "[Terraform] ") &&
        !strcontains(policy.display_name, "[Shadow]") &&
        tolist(policy.notification_channels) == var.notification_channel_ids &&
        policy.user_labels.mode == "paging"
      )
    ])
    error_message = "Every alert policy notifies the configured channels and carries no shadow marker."
  }

  assert {
    condition = alltrue([
      strcontains(google_monitoring_alert_policy.snapshot_feed_stale.conditions[0].condition_matched_log[0].filter, "jsonPayload.operation=\"freshness\""),
      strcontains(google_monitoring_alert_policy.snapshot_feed_stale.conditions[0].condition_matched_log[0].filter, "jsonPayload.outcome=\"stale\""),
      google_monitoring_alert_policy.snapshot_feed_stale.alert_strategy[0].notification_rate_limit[0].period == "3600s",
    ])
    error_message = "The one freshness policy matches the job's own stale verdict and carries no threshold of its own."
  }

  assert {
    condition = alltrue([
      strcontains(google_monitoring_alert_policy.application_errors.conditions[0].condition_matched_log[0].filter, "severity>=ERROR"),
      strcontains(google_monitoring_alert_policy.application_errors.conditions[0].condition_matched_log[0].filter, "NOT logName:\"run.googleapis.com%2Frequests\""),
      strcontains(google_monitoring_alert_policy.application_errors.conditions[0].condition_matched_log[0].filter, "resource.labels.job_name=\"shallweswim-capture\""),
      strcontains(google_monitoring_alert_policy.application_errors.conditions[0].condition_matched_log[0].filter, "resource.labels.service_name=\"shallweswim\""),
      google_monitoring_alert_policy.application_errors.alert_strategy[0].auto_close == "1800s",
      strcontains(google_monitoring_alert_policy.request_5xx.conditions[0].condition_threshold[0].filter, "metric.labels.response_code != \"503\""),
      strcontains(google_monitoring_alert_policy.request_5xx.conditions[0].condition_threshold[0].filter, "run.googleapis.com/request_count"),
    ])
    error_message = "The catch-all policies must cover both application resources, exclude request logs and 503s, and close on their own."
  }

  assert {
    condition = alltrue([
      google_monitoring_uptime_check_config.homepage.http_check[0].path == "/api/healthy?uptime",
      google_monitoring_uptime_check_config.homepage.period == "60s",
      google_monitoring_alert_policy.homepage_uptime_failure.conditions[0].condition_threshold[0].duration == "300s",
      google_monitoring_alert_policy.homepage_uptime_failure.conditions[0].condition_threshold[0].evaluation_missing_data == "EVALUATION_MISSING_DATA_ACTIVE",
    ])
    error_message = "The uptime check probes the health endpoint every minute and its policy fires after five minutes of failure or missing data."
  }

  assert {
    condition = alltrue([
      strcontains(google_monitoring_alert_policy.archive_read_gap.conditions[0].condition_matched_log[0].filter, "jsonPayload.operation=\"hydrate\""),
      strcontains(google_monitoring_alert_policy.archive_read_gap.conditions[0].condition_matched_log[0].filter, "jsonPayload.outcome!=\"success\""),
      strcontains(google_monitoring_alert_policy.archive_read_gap.conditions[0].condition_matched_log[0].filter, "resource.type=\"cloud_run_job\""),
    ])
    error_message = "The archive read gap policy must match every historical read that served fewer years than configured."
  }
}
