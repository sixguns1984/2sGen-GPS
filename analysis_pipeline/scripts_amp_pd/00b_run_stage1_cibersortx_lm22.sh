#!/usr/bin/env bash
set -euo pipefail

ROOT="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt"
LM22=""
ENGINE="docker"
SIF=""
PERM=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --lm22) LM22="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --sif) SIF="$2"; shift 2 ;;
    --perm) PERM="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${CIBERSORTX_EMAIL:?Set CIBERSORTX_EMAIL first}"
: "${CIBERSORTX_TOKEN:?Set CIBERSORTX_TOKEN first}"
[[ -f "$LM22" ]] || { echo "LM22 file not found: $LM22" >&2; exit 2; }

INDIR="$ROOT/data/stage1_manuscript/cell_deconvolution/cibersortx_lm22"
OUTBASE="$INDIR/results"
mkdir -p "$OUTBASE"

declare -A MIX
MIX[PP]="$INDIR/PP_PPMI_LM22_mixture_TPM.tsv"
MIX[PD]="$INDIR/PD_PDBP_LM22_mixture_TPM.tsv"

for COHORT in PP PD; do
  MIXTURE="${MIX[$COHORT]}"
  [[ -f "$MIXTURE" ]] || { echo "Missing Stage-I mixture: $MIXTURE" >&2; exit 2; }
  OUTDIR="$OUTBASE/$COHORT"
  mkdir -p "$OUTDIR"
  cp -f "$LM22" "$OUTDIR/LM22.txt"
  cp -f "$MIXTURE" "$OUTDIR/mixture.tsv"

  echo "===== Stage-I exact CIBERSORTx LM22: $COHORT ====="
  if [[ "$ENGINE" == "docker" ]]; then
    docker run --rm \
      -v "$OUTDIR:/src/data" \
      -v "$OUTDIR:/src/outdir" \
      cibersortx/fractions:latest \
      --username "$CIBERSORTX_EMAIL" \
      --token "$CIBERSORTX_TOKEN" \
      --sigmatrix LM22.txt \
      --mixture mixture.tsv \
      --rmbatchBmode TRUE \
      --perm "$PERM" \
      --QN FALSE \
      --absolute FALSE
  elif [[ "$ENGINE" == "apptainer" || "$ENGINE" == "singularity" ]]; then
    [[ -f "$SIF" ]] || { echo "--sif is required for apptainer/singularity" >&2; exit 2; }
    "$ENGINE" exec \
      --bind "$OUTDIR:/src/data" \
      --bind "$OUTDIR:/src/outdir" \
      "$SIF" \
      /src/CIBERSORTxFractions \
      --username "$CIBERSORTX_EMAIL" \
      --token "$CIBERSORTX_TOKEN" \
      --sigmatrix LM22.txt \
      --mixture mixture.tsv \
      --rmbatchBmode TRUE \
      --perm "$PERM" \
      --QN FALSE \
      --absolute FALSE
  else
    echo "engine must be docker, apptainer, or singularity" >&2
    exit 2
  fi
  echo "Output files in: $OUTDIR"
  find "$OUTDIR" -maxdepth 1 -type f -printf '%f\n' | sort
done
