variable "project_id" {
  description = "GCP project containing the Shall We Swim deployment."
  type        = string
}

variable "service_name" {
  description = "Cloud Run service whose structured events feed the metrics."
  type        = string
  default     = "shallweswim"
}

variable "job_name" {
  description = "Cloud Run Job whose structured events feed the metrics."
  type        = string
  default     = "shallweswim-capture"
}

variable "notification_channel_ids" {
  description = <<-EOT
    Notification channel resource names (projects/PROJECT/notificationChannels/ID)
    that promoted alert policies notify. Channels are created in the console and
    carry the addresses; this module only references them by id, and the ids are
    passed on the plan command line from the operator's environment
    (SHALLWESWIM_ALERT_NOTIFICATION_CHANNELS) so nothing personal enters the
    repository. Empty means no promoted policy notifies anyone.
  EOT
  type        = list(string)
  default     = []
}

variable "site_host" {
  description = "Public hostname the uptime check probes; the application's canonical host."
  type        = string
  default     = "shallweswim.today"
}
