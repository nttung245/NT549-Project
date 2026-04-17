variable "gcp_project" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
}

variable "gcp_svc_key" {
  description = "GCP Service Account Key"
  type        = string
}

variable "instance_name" {
  description = "Name of the GCE instance"
  type        = string
  default     = "rl-training-vm"
}

variable "machine_type" {
  description = "Machine type for the GCE instance"
  type        = string
  default     = "e2-standard-4"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-east1-b"
}
