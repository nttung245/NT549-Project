# GCE Instance for Training
resource "google_compute_instance" "training_vm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 50 # 50GB disk
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral external IP
    }
  }

  metadata_startup_script = file("../scripts/gcp_startup.sh")

  # Service account with necessary scopes
  service_account {
    scopes = ["cloud-platform"]
  }

  tags = ["mlflow", "ssh", "web"]
}

# Firewall rule to allow MLflow traffic
resource "google_compute_firewall" "allow_mlflow" {
  name    = "allow-mlflow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  source_ranges = ["0.0.0.0/0"] # WARNING: Open to all. Consider restricting to your IP.
  target_tags   = ["mlflow"]
}

# Firewall rule to allow SSH traffic via IAP
resource "google_compute_firewall" "allow_ssh_iap" {
  name    = "allow-ssh-iap"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # Allow only Google IAP tunnel IP range
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["ssh"]
}

# Firewall rule to allow HTTP/Web traffic
resource "google_compute_firewall" "allow_web" {
  name    = "allow-web"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["web"]
}

# Bucket to store site/artifacts
resource "google_storage_bucket" "artifacts" {
  name     = "rl-artifacts-${var.gcp_project}"
  location = var.gcp_region
  force_destroy = true
  uniform_bucket_level_access = true
}
