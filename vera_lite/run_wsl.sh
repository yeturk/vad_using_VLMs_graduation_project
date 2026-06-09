#!/usr/bin/env bash
# Helper to run vera_lite scripts inside the grad2_env venv under WSL.
# Usage: wsl -d Ubuntu-22.04 -- bash vera_lite/run_wsl.sh <python-args...>
set -e
cd "$(dirname "$0")/.."
source grad2_env/bin/activate
python "$@"
