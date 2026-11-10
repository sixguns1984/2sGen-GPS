#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TMPDIR=${TMPDIR:-/public/labdata/luojunfeng/tmp}
PC_MODE=${PC_MODE:-eigensoft}
TW_ALPHA=${TW_ALPHA:-0.05}
TWSTATS=${TWSTATS:-twstats}
TWTABLE=${TWTABLE:-twtable}

cd "$CODE"
export TMPDIR TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"

# This prepares the broader/non-strict V8.9 longitudinal resources used by the
# current Stage-II workflow. It is intentionally separate from manuscript-exact Stage-I inputs.
python scripts_amp_pd/00_check_environment.py
python scripts_amp_pd/01_prepare_amp_pd.py \
  --root "$ROOT" \
  --qtl-mean-tpm 1 \
  --deconv-mean-tpm 0.01 \
  --pc-mode "$PC_MODE" \
  --tw-alpha "$TW_ALPHA" \
  --twstats "$TWSTATS" \
  --twtable "$TWTABLE" \
  --plink-threads 16
python scripts_amp_pd/00_audit_after_preprocess.py --root "$ROOT"
python scripts_amp_pd/01e_run_expression_pc_BE.py \
  --root "$ROOT" \
  --groups PP_FULL,PP_LEGACY,PD_LEGACY,PD_FULL,REP_FULL \
  --B 20 --alpha 0.05 --seed 1 --cores 8
python scripts_amp_pd/00_audit_after_preprocess.py --root "$ROOT"
python scripts_amp_pd/02_build_cis_map.py --root "$ROOT" --window 1000000
