# The uptime check and its policy answer "can users reach the site?", the
# one question no log-based metric can, because a site nobody can reach
# writes no logs. Both were created in the console first and imported here,
# so the check keeps its id ("homepage", assigned at creation and read from
# state; the provider does not let a configuration set it) and its history.

resource "google_monitoring_uptime_check_config" "homepage" {
  display_name       = "shallweswim API Healthy Uptime Check"
  period             = "60s"
  timeout            = "30s"
  checker_type       = "STATIC_IP_CHECKERS"
  log_check_failures = true

  http_check {
    path           = "/api/healthy?uptime"
    port           = 80
    use_ssl        = false
    request_method = "GET"

    headers = {
      "X-HealthCheck" = "uptime"
    }

    accepted_response_status_codes {
      status_class = "STATUS_CLASS_2XX"
    }
  }

  monitored_resource {
    type = "uptime_url"

    labels = {
      host       = var.site_host
      project_id = var.project_id
    }
  }
}

resource "google_monitoring_alert_policy" "homepage_uptime_failure" {
  display_name          = "[Terraform] Homepage uptime failure"
  combiner              = "OR"
  enabled               = true
  notification_channels = var.notification_channel_ids
  severity              = "CRITICAL"
  user_labels           = local.paging_alert_labels

  documentation {
    mime_type = "text/markdown"
    content   = "The uptime check on /api/healthy has failed for five minutes from Google's checkers, or stopped reporting. The endpoint answers 200 while any location has data, so this is the site being unreachable or every instance having nothing to serve; check the Cloud Run service's revisions and the web servers' load events (MONITORING.md)."
  }

  conditions {
    display_name = "Failure of uptime check_id ${google_monitoring_uptime_check_config.homepage.uptime_check_id}"

    condition_threshold {
      filter                  = "resource.type = \"uptime_url\" AND metric.type = \"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.labels.check_id = \"${google_monitoring_uptime_check_config.homepage.uptime_check_id}\""
      comparison              = "COMPARISON_GT"
      threshold_value         = 1
      duration                = "300s"
      evaluation_missing_data = "EVALUATION_MISSING_DATA_ACTIVE"

      aggregations {
        alignment_period     = "1200s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.project_id", "resource.label.host"]
      }

      trigger {
        count = 1
      }
    }
  }
}
