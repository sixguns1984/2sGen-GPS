#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=cairo
OUT=$FIGROOT/power_analysis_CTAR_v5

if [[ "${1:-}" == 'plot' ]]; then
  shift
  echo '[PLOT ONLY] Reuse completed CTAR v5 power tables; no CTAR/power rerun.'
  "$PYTHON" -u "$HERE/15_plot_sfigure6_v5.py" --figroot "$FIGROOT" "$@"
  exit 0
fi
OLD=$FIGROOT/power_analysis_empirical_v4
mkdir -p "$OUT" "$FIGROOT/supplementary" "$ROOT/logs"
echo "[1/5] Full Stage1 PPMI -> PDBP independent replication (no refit)"
if [[ -s "$OLD/full_PPMI_PDBP_replication_summary.csv" && "${REUSE_V4:-1}" == 1 ]];then
   cp "$OLD/full_PPMI_PDBP_replication_summary.csv" "$OUT/"
   printf 'Reused previously completed full-cohort Stage1 audit at %s\n' "$OLD" > "$OUT/full_replication_provenance.txt"
else
   "$PYTHON" -u "$HERE/09_full_cohort_replication.py" --root "$ROOT" --figroot "$FIGROOT"
fi
echo "[2/5] Real CTAR effect templates (reuse actual calibrated profiles if possible)"
if [[ -s "$OLD/empirical_CTAR_eight_parameter_effects.csv" && "${REUSE_V4:-1}" == 1 ]];then
   cp "$OLD/empirical_CTAR_eight_parameter_effects.csv" "$OUT/"
   cp "$OLD/calibration_manifest.json" "$OUT/" 2>/dev/null || true
   printf 'Reused actual frozen Stage1 effect calibration from %s\n' "$OLD" > "$OUT/calibration_provenance.txt"
else
   "$PYTHON" -u "$HERE/06_calibrate_real_ctar_effects.py" --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
       --n-calibration "${CALIBRATION_N:-260}" --min-valid 35 --seed "${POWER_SEED:-20261029}"
fi

echo "[3/5] Real PPMI Stage1 validated-IV MAF reference"
if [[ "${MAF_REFERENCE:-all_eligible_IVs}" == "calibration" ]];then
   "$PYTHON" - "$OUT" <<'PY'
import sys,pandas as pd
from pathlib import Path
out=Path(sys.argv[1]);z=pd.read_csv(out/'empirical_CTAR_eight_parameter_effects.csv')
z[['lead_snp','MAF_observed','n_participants_called']].rename(columns={'MAF_observed':'MAF','n_participants_called':'n_called'}).to_csv(out/'MAF_frozen_Stage1_IVs.csv',index=False)
print('[MAF] FAST option: calibrated SNPs only. Main reference median is not full IV-set median.')
PY
else
   "$PYTHON" -u "$HERE/13_stage1_maf_reference.py" --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
     --chunk "${MAF_BATCH:-400}"
fi

echo "[4/5] GLS-standardized signal scenarios with real CTAR trajectories; vector plot"
"$PYTHON" -u "$HERE/14_power_standardized_pve_v5.py" --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
  --mafs "${POWER_MAFS:-0.05,0.10,0.20,0.30}" \
  --pve-levels "${POWER_R2_LEVELS:-0.005,0.01,0.02,0.05}" \
  --n-grid "${POWER_N_GRID:-100,150,200,250,300,400,500,700,900,1200,1373,1405,1600,2000,2600,3500,5000,7000}" \
  --rho "${POWER_RHO:-auto}" --reference-maf "${POWER_REFERENCE_MAF:-auto}" \
  --n-shapes "${POWER_N_SHAPES:-24}" --draws "${POWER_GENO_DRAWS:-3000}" \
  --seed "${POWER_SEED:-20261029}" --profile "${POWER_PROFILE:-PD}"

if [[ "${RUN_SUBSAMPLING:-0}" == "1" ]];then
   echo '[4b/5] Expensive optional donor subsampling with production CTAR'
   "$PYTHON" -u "$HERE/10_ctar_subsample_existing_model.py" \
        --root "$ROOT" --out-root "$FIGROOT/v5_subsampling" --n-pairs "${SUBSAMPLE_N_PAIRS:-300}" \
        --resamples "${SUBSAMPLE_REPS:-8}" --fractions "${SUBSAMPLE_FRACTIONS:-0.5,0.75,0.9,1.0}" \
        --rho-points "${RHO_POINTS:-100}" --device "${DEVICE:-cpu}" --seed "${POWER_SEED:-20261029}"
   "$PYTHON" -u "$HERE/11_summarize_subsamples.py" --root "$ROOT" --out-root "$FIGROOT/v5_subsampling"
elif [[ "${REUSE_V4_SUBSAMPLING:-1}" == "1" && -s "$FIGROOT/v4_subsampling/power_analysis_independent_v2/PDBP_replication_vs_sample_size.csv" ]];then
    mkdir -p "$FIGROOT/v5_subsampling/power_analysis_independent_v2"
    cp "$FIGROOT/v4_subsampling/power_analysis_independent_v2/PDBP_replication_vs_sample_size.csv" \
       "$FIGROOT/v5_subsampling/power_analysis_independent_v2/"
    echo 'V5 reused v4 random PPMI candidate subsampling with its original fixed candidate set and no re-fitting.' \
      > "$FIGROOT/v5_subsampling/reuse_provenance.txt"
    echo '[SUBSAMPLE] Reused v4 CTAR refit results (distinct from all-pair full cohort statistic)'
else
    echo '[SUBSAMPLE] Skipped; panel a will show all-pair complete-cohort replication only.'
fi

echo '[5/5] Supplementary Figure 6 vector PDF / SVG'
"$PYTHON" -u "$HERE/15_plot_sfigure6_v5.py" --figroot "$FIGROOT"
echo '[COMPLETE] CTAR v5 Supplementary Figure 6'
