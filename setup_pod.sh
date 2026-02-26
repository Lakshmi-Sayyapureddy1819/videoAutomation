#!/bin/bash

# VidRush Cloud Worker Setup Script
# Run this on your RunPod terminal to install VidGear, FFmpeg, and other dependencies.

set -e

echo "🚀 Setting up VidRush Environment..."

# 1. Install System Dependencies (FFmpeg, OpenGL for OpenCV)
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx

# 2. Install Python Dependencies
echo "📦 Installing Python Libraries..."
pip install --upgrade pip
pip install -r src/requirements.txt

echo "✅ Setup Complete! You can now run: python src/pod_server.py"