#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$CODE"

# Run sequentially by default so the two methods do not compete for the same GPUs.
bash "$CODE/run_stage1_ctar_replication.sh"
bash "$CODE/run_stage1_linreg_cov_replication.sh"

python scripts_amp_pd/05_summarize_replication.py --root "$ROOT"
