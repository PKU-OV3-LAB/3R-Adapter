#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
exec python "$SCRIPT_DIR/correspondence.py" "$@"
