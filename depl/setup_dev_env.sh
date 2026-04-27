#!/usr/bin/env bash
set -euo pipefail

# Create a Python virtual environment named DEV in this project directory.
python3 -m venv DEV

# Activate the environment and install dependencies.
# For macOS / Linux:
source DEV/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Development environment 'DEV' created and dependencies installed."
echo "Activate it with: source DEV/bin/activate"
