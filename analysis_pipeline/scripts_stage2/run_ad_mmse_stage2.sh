#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
CODE="${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}"
ADBASE="${ADBASE:-/public2/data/luojunfeng/2sTGCS_binary/outcome/AD}"
ANALYSIS_NAME="${ANALYSIS_NAME:-ad_mmse}"
BASE="$ROOT/single_iv_causal/$ANALYSIS_NAME"
PREP="$ROOT/single_iv_causal/ad_mmse/prep"
CANDIDATES="${CANDIDATES:-$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv}"
SOURCE_BIM="${SOURCE_BIM:-/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged.bim}"
MAP_BFILE="${MAP_BFILE:-/public/labdata/public_data/AMP-AD/ROSMAP/Topmed_imputation/Topmed_out/data/chr1_22.dose}"
ADNI_BFILE="${ADNI_BFILE:-/public/labdata/public_data/ADNI/ADNI_WGS/ADNI_WGS_GATK/ADNI.808_indiv.WGS}"
MAP_COV="${MAP_COV:-$ADBASE/ROSMAP/2sggps_impute/mmse.cov.csv}"
# ADNI phenotype remains in the earlier directory; do not use a WGS phenotype file.
ADNI_COV="${ADNI_COV:-$ADBASE/ADNI/2sggps/adni_mapimpute_cover/mmse.cov.csv}"
MAP_PC_BFILE="${MAP_PC_BFILE:-$MAP_BFILE}"
ADNI_PC_BFILE="${ADNI_PC_BFILE:-$ADNI_BFILE}"
PLINK="${PLINK:-plink1.9}"
N_JOBS="${N_JOBS:-12}"
MIN_PARTICIPANTS="${MIN_PARTICIPANTS:-50}"
MAX_LAG="${MAX_LAG:-4}"
AR_MODE="${AR_MODE:-off}"
mkdir -p "$PREP" "$BASE"
command -v "$PLINK" >/dev/null || { echo "Missing PLINK binary: $PLINK" >&2; exit 127; }
PLINK="$(command -v "$PLINK")"
[[ -x "$PLINK" ]] || { echo "PLINK not executable: $PLINK" >&2; exit 126; }
echo "[TOOLS] PLINK=$PLINK"
"$PLINK" --version
for prefix in "$MAP_BFILE" "$ADNI_BFILE"; do
  for ext in .bed .bim .fam; do
    [[ -s "$prefix$ext" ]] || { echo "Missing genotype input: $prefix$ext" >&2; exit 1; }
  done
done
[[ -s "$MAP_COV" && -s "$ADNI_COV" ]] || { echo "Missing phenotype cov file" >&2; exit 1; }
echo "[INPUT] MAP genotype=$MAP_BFILE (full TOPMed)"
echo "[INPUT] MAP cov=$MAP_COV (unchanged)"
echo "[INPUT] ADNI genotype=$ADNI_BFILE (full WGS)"
echo "[INPUT] ADNI cov=$ADNI_COV (unchanged)"
echo "[INPUT] PC sources: MAP=$MAP_PC_BFILE ADNI=$ADNI_PC_BFILE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
COV="$PREP/AD_MMSE_phenotype_cov.csv"
CAND="$PREP/AD_MMSE_shared_frozen_candidates.csv"
if [[ "${SKIP_AD_STEPS_1_3:-0}" == "1" ]]; then
  echo '[WARNING] SKIP_AD_STEPS_1_3=1: reuse only if candidates and sources are independently verified; this option is NOT recommended after a SNP matching change'
  python - "$PREP/preparation_manifest.json" "$MAP_BFILE" "$ADNI_BFILE" <<'PY_CHECK_MANIFEST'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); expected={'MAP':sys.argv[2],'ADNI':sys.argv[3]}
if not p.exists(): raise SystemExit(f'Cannot skip steps 1–3; no manifest: {p}')
m=json.loads(p.read_text()); observed={r['cohort']:r['bfile'] for r in m.get('cohorts',[])}
for cohort in expected:
    if observed.get(cohort)!=expected[cohort]:
        raise SystemExit(f'Stale genotype prep for {cohort}: prior={observed.get(cohort)!r}, requested={expected[cohort]!r}. Run SKIP_AD_STEPS_1_3=0.')
print('[CACHE VALIDATION] Input genotype prefixes match existing prep manifest; reusing Steps 1–3',flush=True)
PY_CHECK_MANIFEST
  [[ -s "$COV" && -s "$CAND" && -s "$BASE/genotype/single_iv_snps.bed" ]] || { echo 'Missing required Step 1–3 artifacts' >&2; exit 1; }
else
  echo '[STEP 1] MAP full TOPMed + ADNI full WGS SNP matching; retain both MMSE cov files'
  python "$CODE/scripts_stage2/16_prepare_ad_mmse.py" \
    --root "$ROOT" --map-cov "$MAP_COV" --adni-cov "$ADNI_COV" \
    --map-bfile "$MAP_BFILE" --adni-bfile "$ADNI_BFILE" \
    --source-bim "$SOURCE_BIM" --candidate-file "$CANDIDATES" --plink "$PLINK"
  echo '[STEP 2] Build fresh merged bfile from INTERSECTION of full TOPMed MAP and full WGS ADNI IVs'
  COMMON="$PREP/MAP_ADNI_common_IVs.txt"
  comm -12 \
     <(awk '{print $2}' "$PREP/MAP_harmonized_A1.bim" | LC_ALL=C sort -u) \
     <(awk '{print $2}' "$PREP/ADNI_harmonized_A1.bim" | LC_ALL=C sort -u) > "$COMMON"
  n_common="$(wc -l < "$COMMON")"
  echo "[COMMON IVs] $n_common"
  [[ "$n_common" -gt 0 ]] || { echo 'No shared harmonized IVs; check genome build and alleles' >&2; exit 1; }
  for cohort in MAP ADNI; do
    "$PLINK" --bfile "$PREP/${cohort}_harmonized_A1" --extract "$COMMON" \
      --keep-allele-order --make-bed --out "$PREP/${cohort}_common_IVs"
  done
  MERGED="$PREP/MAP_ADNI_harmonized_merged"
  # ALWAYS rebuild: previous merged files may originate from old filtered MAP/ADNI data.
  rm -f "$MERGED.bed" "$MERGED.bim" "$MERGED.fam"
  "$PLINK" --bfile "$PREP/MAP_common_IVs" \
    --bmerge "$PREP/ADNI_common_IVs.bed" "$PREP/ADNI_common_IVs.bim" "$PREP/ADNI_common_IVs.fam" \
    --keep-allele-order --make-bed --out "$MERGED"
  n_merged="$(wc -l < "$MERGED.bim")"
  echo "[MERGED] IVs=$n_merged; expected=$n_common"
  [[ "$n_merged" -eq "$n_common" ]] || { echo 'Merged variants differ from shared IVs; abort' >&2; exit 1; }
  echo '[STEP 3] Extract validated frozen IVs ONLY; NO --mind/--geno/--maf/--hwe/LD-pruning; preserve A1 dosage direction'
  python "$CODE/scripts_stage2/19_prepare_ad_mmse_iv_no_reqc.py" \
    --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --phenotype-cov "$COV" \
    --candidate-file "$CAND" --raw-bfile "$MERGED" --plink "$PLINK"
fi

echo '[STEP 4] MAP & ADNI separate ancestry PCs from high-density original genotypes'
if [[ "${SKIP_AD_PCS:-0}" == "1" ]]; then
  python - "$BASE/genotype_pcs/pc_selection.json" "$BASE/genotype_pcs/single_iv_genotype_pcs.csv" \
      "$PREP/AD_MMSE_phenotype_cov.csv" "$MAP_PC_BFILE" "$ADNI_PC_BFILE" "${MINIMUM_PCS:-5}" <<'PY_PC_CACHE'
import json,sys,pathlib,pandas as pd
manifest_path,pcs_path,cov_path,mapbf,adnibf,minpcs=sys.argv[1:]
for f in (manifest_path,pcs_path,cov_path):
    if not pathlib.Path(f).is_file():raise SystemExit(f'Cannot skip PCA: missing {f}')
manifest=json.loads(pathlib.Path(manifest_path).read_text())
expected={'MAP':mapbf,'ADNI':adnibf}
for cohort,bfile in expected.items():
    entry=manifest.get('cohorts',{}).get(cohort,{})
    if entry.get('genomewide_bfile') != bfile:
        raise SystemExit(f'Cannot skip PCA: {cohort} source changed. Refuse stale PCs.')
    if len(entry.get('selected_columns',[])) < int(minpcs):
        raise SystemExit(f'Cannot skip PCA: {cohort} does not meet selected minimum PCs {minpcs}')
co=pd.read_csv(cov_path,usecols=['cohort','participant_id'],dtype=str).drop_duplicates()
pc=pd.read_csv(pcs_path,dtype={'cohort':str,'participant_id':str})
for cohort in expected:
    required=set(co.loc[co.cohort.eq(cohort),'participant_id'])
    existing=set(pc.loc[pc.cohort.eq(cohort),'participant_id'])
    if not required.issubset(existing):
        raise SystemExit(f'Cannot skip PCA: {cohort} missing {len(required-existing)} phenotype participants')
    for col in manifest['cohorts'][cohort]['selected_columns']:
        if col not in pc.columns:raise SystemExit(f'Cannot skip PCA: {col} not in PCs')
print('[PCA CACHE VALID] Sources, participant coverage, selected PC columns verified; reuse existing genome-wide PCs',flush=True)
PY_PC_CACHE
else
  python "$CODE/scripts_stage2/04_prepare_ad_mmse_genomewide_pcs.py" \
    --root "$ROOT" --analysis-name "$ANALYSIS_NAME" \
    --map-genome-bfile "$MAP_PC_BFILE" --adni-genome-bfile "$ADNI_PC_BFILE" \
    --plink "$PLINK" --minimum-pcs "${MINIMUM_PCS:-5}"
fi

echo '[STEP 5] Re-predict FROZEN Stage1 GReX at every observed MMSE time (no time-support cutoff)'
python "$CODE/scripts_stage2/18_materialize_ad_mmse_all_followup.py" \
  --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --phenotype-cov "$COV" \
  --candidate-file "$CAND" --single-iv-bfile "$BASE/genotype/single_iv_snps" \
  --n-jobs "$N_JOBS" --plink "$PLINK"

echo '[STEP 6] MAP discovery FDR; ADNI same-lag, same-direction replication'
python "$CODE/scripts_stage2/17_run_ad_mmse_temporal_fast.py" \
  --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --phenotype-cov "$COV" \
  --outcome-col 'mmse' --candidate-file "$CAND" \
  --discovery-cohort MAP --replication-cohorts ADNI --replication-pooled-groups '' \
  --max-lag "$MAX_LAG" --min-participants "$MIN_PARTICIPANTS" \
  --fdr-alpha 0.05 --time-varying-covariates '' \
  --use-genotype-pcs --adjust-current-age-cubic \
  --no-require-stage1-followup-support --ar-mode "$AR_MODE" --detail-mode minimal \
  --n-jobs "$N_JOBS" --no-resume-discovery --no-resume-replication \
  --out-dir "$BASE/temporal_causal_agecubic_full_followup_fast"

echo '[DONE] Outputs:' "$BASE/temporal_causal_agecubic_full_followup_fast"
