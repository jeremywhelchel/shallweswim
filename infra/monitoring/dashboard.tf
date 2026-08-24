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
            title = "Feed update duration p95 by location/feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.location} / $${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.feed_update_duration.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.location", "metric.label.feed"]
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
            title = "Plot availability latency p95 by location/feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.location} / $${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.plot_availability_latency.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MAX"
                      groupByFields      = ["metric.label.location", "metric.label.feed"]
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
            title = "Published feed record count p50 by location/feed"
            xyChart = {
              dataSets = [{
                plotType       = "LINE"
                targetAxis     = "Y1"
                legendTemplate = "$${metric.labels.location} / $${metric.labels.feed}"
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"${local.metric_prefix}/${google_logging_metric.feed_records.name}\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_PERCENTILE_50"
                      crossSeriesReducer = "REDUCE_MIN"
                      groupByFields      = ["metric.label.location", "metric.label.feed"]
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
        }
      ]
    }
  })
}
