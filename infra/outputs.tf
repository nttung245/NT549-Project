output "instance_external_ip" {
  value       = google_compute_instance.training_vm.network_interface[0].access_config[0].nat_ip
  description = "The public IP address of the training VM. Use this to connect via standard SSH."
}
