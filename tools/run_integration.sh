#!/usr/bin/env zsh
# Run integration tests in a developer conda environment.
# Usage:
#   ./tools/run_integration.sh [--recreate]
#
# The script will:
#  - generate environment-dev.yml via the Python helper
#  - create pyparticle-dev if missing, or update it
#  - run pytest on the integration tests inside the pyparticle-dev env

set -euo pipefail

RECREATE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --recreate)
      RECREATE=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Activate conda or install Miniforge/Anaconda." >&2
  exit 2
fi

PYGEN="$ROOT/tools/generate_env_dev.py"
if [[ ! -f "$PYGEN" ]]; then
  echo "Generator script not found: $PYGEN" >&2
  exit 2
fi

echo "Generating environment-dev.yml..."
conda run -n pyparticle --no-capture-output python "$PYGEN" || python "$PYGEN"

ENVFILE="$ROOT/environment-dev.yml"
if [[ ! -f "$ENVFILE" ]]; then
  echo "environment-dev.yml not found after generation" >&2
  exit 3
fi

if [[ $RECREATE -eq 1 ]]; then
  echo "Recreating conda env pyparticle-dev (will remove if present)..."
  if conda env list | awk '{print $1}' | grep -q "^pyparticle-dev$"; then
    echo "Removing existing environment pyparticle-dev..."
    conda env remove -n pyparticle-dev || true
  fi
  echo "Creating pyparticle-dev from $ENVFILE..."
  conda env create -f "$ENVFILE"
else
  if conda env list | awk '{print $1}' | grep -q "^pyparticle-dev$"; then
    echo "Updating existing pyparticle-dev from $ENVFILE..."
    conda env update -n pyparticle-dev -f "$ENVFILE" || true
  else
    echo "Creating pyparticle-dev from $ENVFILE..."
    conda env create -f "$ENVFILE"
  fi
fi

echo "Running integration tests inside pyparticle-dev..."
conda run -n pyparticle-dev pytest tests/integration -q -rA

echo "Integration test run complete."
