#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=cairo
OUT=$FIGROOT/power_analysis_CTAR_v6

# Plot-only entrypoint: deliberately exits BEFORE calibration, replication, or power computations.
if [[ "${1:-}" == 'plot' ]]; then
  shift
  echo '[PLOT ONLY] Reuse completed CTAR v6 numerical results; no CTAR/power rerun.'
  "$PYTHON" -u "$HERE/22_plot_sfigure6_v6.py" --figroot "$FIGROOT" "$@"
  exit 0
fi
mkdir -p "$OUT" "$FIGROOT/supplementary" "$ROOT/logs"

# Hotfix v6.1: v6.0 calibration mistakenly saved in the v5 directory.
# Migrate only when the correct destination is absent. We validate the 8-parameter
# source schema and never mutate the original v5 files.
LEGACY_CAL=$FIGROOT/power_analysis_CTAR_v5
if [[ ! -s "$OUT/calibrated_real_full8_CTAR_PVE.csv" && -s "$LEGACY_CAL/calibrated_real_full8_CTAR_PVE.csv" ]]; then
  echo "[RECOVERY] Detected v6 calibration saved under v5: $LEGACY_CAL"
  "$PYTHON" - "$LEGACY_CAL" "$OUT" <<'PYMIGRATE'
import csv
import shutil
import sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
calib = src / 'calibrated_real_full8_CTAR_PVE.csv'
with calib.open(newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    required = {'gene', 'lead_snp', 'signal_fraction', 'residual_sd',
                'MAF_observed', 'n_hom_ref', 'n_het', 'n_hom_alt'} | {
                f'beta_g_{i}' for i in range(8)}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise SystemExit(f'[RECOVERY] Refusing incompatible calibration: missing {sorted(missing)}')
    n = sum(1 for _ in reader)
    if n < 5:
        raise SystemExit(f'[RECOVERY] Refusing calibration with only {n} rows')
for name in ('calibrated_real_full8_CTAR_PVE.csv', 'calibration_exclusions.csv',
             'calibration_manifest.json', 'three_real_CTAR_curve_templates_audit.csv'):
    source = src / name
    target = dst / name
    if source.is_file() and not target.exists():
        shutil.copy2(source, target)
        print(f'[RECOVERY] Copied {name}: {source} -> {target}', flush=True)
print(f'[RECOVERY] Reused {n} valid 8-parameter calibration rows; no recalibration.', flush=True)
PYMIGRATE
fi

echo '[1/4] Actual PPMI full 8-param PVE calibration: real genotype, RNA expression, visit times'
if [[ -s "$OUT/calibrated_real_full8_CTAR_PVE.csv" && "${RECALIBRATE:-0}" != 1 ]]; then
  echo "[REUSE] $OUT/calibrated_real_full8_CTAR_PVE.csv (RECALIBRATE=1 to refresh)"
else
  "$PYTHON" -u "$HERE/20_calibrate_actual_pve_v6.py" \
    --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
    --n-calibration "${CALIBRATION_N:-400}" --min-valid "${MIN_VALID:-30}" \
    --batch "${CALIBRATION_BATCH:-25}" --seed "${POWER_SEED:-20261029}"
fi

echo '[2/4] All matched PPMI-discovered CTAR -> PDBP replication summary, no CTAR rerun'
if [[ -s "$FIGROOT/power_analysis_CTAR_v5/full_PPMI_PDBP_replication_summary.csv" ]];then
  cp "$FIGROOT/power_analysis_CTAR_v5/full_PPMI_PDBP_replication_summary.csv" "$OUT/full_PPMI_PDBP_replication_summary.csv"
elif [[ -s "$FIGROOT/power_analysis_empirical_v4/full_PPMI_PDBP_replication_summary.csv" ]];then
  cp "$FIGROOT/power_analysis_empirical_v4/full_PPMI_PDBP_replication_summary.csv" "$OUT/full_PPMI_PDBP_replication_summary.csv"
else
  "$PYTHON" -u "$HERE/09_full_cohort_replication.py" --root "$ROOT" --figroot "$FIGROOT"
  cp "$FIGROOT/power_analysis_CTAR_v5/full_PPMI_PDBP_replication_summary.csv" "$OUT/"
fi

echo '[3/4] Fast conditional 8-df planning from empirical whole-time genetic trajectories; no new CTAR fits'
"$PYTHON" -u "$HERE/21_power_empirical_full8_v6.py" \
  --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
  --n-grid "${POWER_N_GRID:-70,100,150,200,300,400,500,700,900,1100,1200,1373,1405,1600,2000,2600,3500,5000,7000}" \
  --draws "${POWER_GENO_DRAWS:-600}" --rho "${POWER_RHO:-auto}" \
  --effect-multipliers "${POWER_MULTIPLIERS:-1.0,0.75,0.5}" \
  --seed "${POWER_SEED:-20261029}" \
  ${POWER_HWE:+--hwe}

echo '[4/4] Write vector PDF / SVG Supplementary Fig. S6 and numeric audit'
"$PYTHON" -u "$HERE/22_plot_sfigure6_v6.py" --figroot "$FIGROOT"
echo "[DONE] $FIGROOT/supplementary/Supplementary_Figure_6_CTAR_empirical_full8_power_v6.pdf"
