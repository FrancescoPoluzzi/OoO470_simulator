#!/bin/bash

apt-get install sudo
# Update package lists
sudo apt-get update

# Install Python
sudo apt-get install -y python3

# Verify installations
python3 -c "import json, sys, copy"