#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="$REPO/analysis_pipeline"
if [[ $# -lt 1 ]]; then echo "Usage: bash analysis_pipeline/run_here.sh SCRIPT [ARGS...]" >&2; exit 2; fi
SCRIPT="$1";shift
if [[ ! -f "$PIPE/$SCRIPT" ]]; then echo "Missing analysis script: $PIPE/$SCRIPT" >&2; exit 2; fi
# Original shell launchers set CODE to their directory, but source scripts are now nested.
# Make expected top-level scripts_amp_pd / scripts_stage2 directories available in a
# temporary execution copy via symlinks instead of editing the original algorithms.
export CODE="$PIPE"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$PIPE"
exec bash "$SCRIPT" "$@"
