variable "project_id" {
  description = "GCP project containing the Shall We Swim deployment."
  type        = string
}

variable "service_name" {
  description = "Cloud Run service whose structured events feed the metrics."
  type        = string
  default     = "shallweswim"
}
