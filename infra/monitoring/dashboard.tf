locals {
  metric_prefix = "logging.googleapis.com/user"
}

resource "google_monitoring_dashboard" "operations" {
  dashboard_json = jsonencode({
    displayName = "Shall We Swim Operations [Terraform]"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Feed updates per 5 minutes by outcome"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.outcome}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.feed_updates.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.outcome"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "updates / 5 min"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Feed update duration p95 by feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.feed_update_duration.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.feed"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "ms"
                scale = "LOG10"
              }
            }
          }
        },
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Plot completions per 5 minutes by outcome"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.outcome}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.plot_generations.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.outcome"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "plots / 5 min"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Plot availability latency p95 by feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.plot_availability_latency.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.feed"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "ms"
                scale = "LOG10"
              }
            }
          }
        },
        {
          yPos   = 8
          width  = 12
          height = 4
          widget = {
            title = "Published feed record count p50 by feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.feed_records.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_50"
                      crossSeriesReducer = "REDUCE_MIN"
                      groupByFields      = ["metric.label.feed"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "records"
                scale = "LOG10"
              }
            }
          }
        },
        {
          yPos   = 12
          width  = 12
          height = 4
          widget = {
            title = "Archive merges per hour by outcome"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.outcome}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.archive_merges.name}\""
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.outcome"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "merges / hour"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          yPos   = 16
          width  = 6
          height = 4
          widget = {
            title = "Capture runs per hour by outcome"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.outcome}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.updater_runs.name}\""
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.outcome"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "runs / hour"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 16
          width  = 6
          height = 4
          widget = {
            title = "Snapshot publishes per hour by outcome"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.outcome}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.snapshot_publishes.name}\""
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields      = ["metric.label.outcome"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "publishes / hour"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          yPos   = 20
          width  = 6
          height = 4
          widget = {
            title = "Archive merge duration p95 by source per hour"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.source}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.archive_merge_duration.name}\""
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.source"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "ms"
                scale = "LOG10"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 20
          width  = 6
          height = 4
          widget = {
            title = "New observations per hour by source (estimated)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.source}"
                timeSeriesQuery = {
                  # The row-count metrics are DISTRIBUTION valued, so the
                  # timeSeriesFilter widget cannot sum them. MQL's sum_from()
                  # reduces each distribution point to its estimated sum, which
                  # the hourly delta then totals per source.
                  timeSeriesQueryLanguage = <<-EOT
                    fetch cloud_run_job
                    | metric '${local.metric_prefix}/${google_logging_metric.archive_merge_new_rows.name}'
                    | align delta(1h)
                    | every 1h
                    | group_by [source: metric.source], [rows: sum(sum_from(val()))]
                  EOT
                }
              }]
              yAxis = {
                label = "observations / hour"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 24
          width  = 6
          height = 4
          widget = {
            title = "Snapshot feed age max by feed per hour"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.snapshot_feed_age.name}\""
                    # One hourly publish per feed and location makes the hour's
                    # 99th percentile that hour's maximum published age.
                    aggregation = {
                      alignmentPeriod    = "3600s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_99"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.feed"]
                    }
                  }
                }
              }]
              yAxis = {
                label = "seconds"
                scale = "LOG10"
              }
            }
          }
        },
        {
          yPos   = 24
          width  = 6
          height = 4
          widget = {
            title = "Revised observations per hour by source (estimated)"
            xyChart = {
              dataSets = [{
                plotType       = "STACKED_BAR"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.source}"
                timeSeriesQuery = {
                  timeSeriesQueryLanguage = <<-EOT
                    fetch cloud_run_job
                    | metric '${local.metric_prefix}/${google_logging_metric.archive_merge_revised_rows.name}'
                    | align delta(1h)
                    | every 1h
                    | group_by [source: metric.source], [rows: sum(sum_from(val()))]
                  EOT
                }
              }]
              yAxis = {
                label = "observations / hour"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}
