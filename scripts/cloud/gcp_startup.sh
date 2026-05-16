#!/bin/bash
# GCP Startup Script for Aircraft RL Project
# This script installs Docker and prepares the environment

set -e

echo "🚀 Starting GCP Setup..."

# Update and install dependencies
apt-get update
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git

# Install Docker
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Setup project directory
mkdir -p /opt/rl-project
cd /opt/rl-project

# Note: The actual cloning will be handled in the Docker execution flow
# but we ensure Docker is running and ready.
systemctl enable docker
systemctl start docker

echo "✅ Docker installed and ready."

# Setup MLflow and project directory permissions
mkdir -p /opt/rl-project/mlruns
chmod -R 777 /opt/rl-project

echo "🎉 Startup script completed successfully."
