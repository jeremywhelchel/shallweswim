terraform {
  required_version = ">= 1.7.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Supply the private, versioned state bucket during init:
  # terraform init -backend-config="bucket=BUCKET_NAME" -backend-config="prefix=monitoring"
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
}
