#!/usr/bin/env python3
"""Consolidated PPMI -> PDBP Stage-I replication summary.

All Stage-I methods with an explicit genotype-by-time interaction test are compared
under one unified CTAR-style dynamic-QTL definition:

  1. BH across the method's full finite temporal/overall test family (G).
  2. Let R be the number of temporal tQTLs passing BH-FDR < alpha.
  3. Confirm the single genotype-by-time interaction hypothesis at
     alpha_II = alpha * R / G.
  4. dynamic tQTL = temporal tQTL + confirmed interaction.

The numeric alpha_II is method-specific because R/G is method-specific; the
*procedure* is identical across methods.  Methods without an explicit interaction
test (AR, union, Fisher, MANOVA) remain temporal-tQTL comparators only.

PDBP replication is independent and nominal (temporal P<0.05, and for dynamic
replication also interaction P<0.05).  PDBP never redefines the PPMI discovery set.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except ImportError:  # clearer runtime failure; pyarrow is listed in requirements.txt
    pq = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import (
    AmpPdPaths,
    FOLLOWUP_METHODS,
    REVISED_FOLLOWUP_METHODS,
    ALL_FOLLOWUP_METHODS,
    implementation_tag,
)
from gengps.utils import storey_pi1

KEYS = ["Variant_ID", "Ensembl_ID"]

DYNAMIC_CAPABLE_METHODS = {
    "submitted_followup_mptga",
    "submitted_followup_ctar_mptga",
    "lin2018_mptga",
    "lin2018_regression",
    "lin2018_regression_cov",
}
UNIFIED_DYNAMIC_POLICY = "stagewise_hierarchical"


def _parquet_file(path):
    if pq is None:
        raise RuntimeError(
            "pyarrow is required for Stage-I parquet summaries. "
            "Install the environment from requirements.txt before running this script."
        )
    return pq.ParquetFile(path)


def method_data_mode(method: str) -> str:
    if method in REVISED_FOLLOWUP_METHODS:
        return "current_v89"
    if method in FOLLOWUP_METHODS:
        return "stage1_manuscript"
    raise ValueError(f"Unknown Stage-I follow-up method: {method}")


def _assert_implementation(path, method, data_mode):
    pf = _parquet_file(path)
    if "implementation_tag" not in pf.schema.names or pf.metadata.num_rows == 0:
        raise RuntimeError(f"{path} lacks an implementation_tag; do not mix stale Stage-I results")
    got = pf.read_row_group(0, columns=["implementation_tag"]).column(0)[0].as_py()
    exp = implementation_tag(method, data_mode)
    if exp is not None and str(got) != str(exp):
        raise RuntimeError(
            f"{path} implementation_tag={got!r}, expected {exp!r} for data_mode={data_mode}. "
            "Re-run/re-finalize Stage-I rather than summarizing mixed results."
        )


def _unique_or_fail(df, label):
    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        ex = df.loc[dup, KEYS].value_counts().head(10)
        raise RuntimeError(f"{label} contains duplicated SNP-gene pairs; examples:\n{ex}")


def _read(path, need_interaction=False):
    schema = set(_parquet_file(path).schema.names)
    cols = KEYS + ["Pvalue"]
    if "FDR" in schema:
        cols += ["FDR"]
    if need_interaction and "Interaction_Pvalue" in schema:
        cols += ["Interaction_Pvalue"]
    if need_interaction and "Interaction_FDR" in schema:
        cols += ["Interaction_FDR"]
    z = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
    _unique_or_fail(z, str(path))
    return z


def _read_replication(path):
    schema = set(_parquet_file(path).schema.names)
    cols = KEYS + ["Pvalue"]
    if "Interaction_Pvalue" in schema:
        cols += ["Interaction_Pvalue"]
    z = pd.read_parquet(path, columns=cols)
    _unique_or_fail(z, str(path))
    return z


def _merge(d, r):
    return d.merge(r, on=KEYS, how="inner", suffixes=("_dis", "_rep"), validate="one_to_one")


def _spearman_logp(m, pdis="Pvalue_dis", prep="Pvalue_rep"):
    if len(m) < 3 or pdis not in m.columns or prep not in m.columns:
        return np.nan
    x = -np.log10(np.clip(pd.to_numeric(m[pdis], errors="coerce"), 1e-300, 1))
    y = -np.log10(np.clip(pd.to_numeric(m[prep], errors="coerce"), 1e-300, 1))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))


def _safe_pi1(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(storey_pi1(x)) if len(x) else np.nan


def _fdr_metadata(final_dir: Path, method: str):
    info_path = final_dir / "fdr_finalize_summary.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    strategy = str(info.get("fdr_strategy", "unknown"))
    alpha = float(info.get("alpha", info.get("fdr_alpha", 0.05)))
    G = info.get("hierarchical_screening_n_tests_G", info.get("G", info.get("n_global_tests")))
    R = info.get("hierarchical_screening_n_rejected_R", info.get("R", info.get("n_screened_tqtl")))
    alpha_ii = info.get("hierarchical_confirmation_alpha", info.get("alpha_II"))
    if method not in DYNAMIC_CAPABLE_METHODS:
        policy = "temporal global BH; no explicit interaction test for this comparator"
    elif strategy == "stagewise_hierarchical":
        policy = "temporal global BH; interaction confirmation alpha_II = alpha * R / G"
    elif strategy == "global_global":
        policy = "LEGACY SENSITIVITY ONLY: temporal global BH; interaction global BH"
    else:
        policy = "unknown/unfinalized dynamic policy"
    return {
        "fdr_strategy": strategy,
        "fdr_alpha": alpha,
        "discovery_fdr_policy": policy,
        "discovery_test_family_G": G,
        "discovery_screened_R": R,
        "interaction_confirmation_alpha": alpha_ii,
        "dynamic_policy_applicable": bool(method in DYNAMIC_CAPABLE_METHODS),
        "dynamic_policy_comparable": bool(
            method not in DYNAMIC_CAPABLE_METHODS or strategy == UNIFIED_DYNAMIC_POLICY
        ),
    }


def summarize_label(paths, method, label=None, data_mode=None, require_unified_dynamic_policy=True):
    label = str(label or method)
    data_mode = str(data_mode or method_data_mode(method))
    pp = paths.qtl_dir / "PP" / label / "final"
    pd_dir = paths.qtl_dir / "PD" / label / "final"
    tqtl_file = pp / "significant_tqtl.parquet"
    if not tqtl_file.exists():
        tqtl_file = pp / "significant_temporal.parquet"
    rep_file = pd_dir / "all_results.parquet"
    if not tqtl_file.exists() or not rep_file.exists():
        return None

    _assert_implementation(tqtl_file, method, data_mode)
    _assert_implementation(rep_file, method, data_mode)

    fdr_meta = _fdr_metadata(pp, method)
    if (
        require_unified_dynamic_policy
        and method in DYNAMIC_CAPABLE_METHODS
        and fdr_meta["fdr_strategy"] != UNIFIED_DYNAMIC_POLICY
    ):
        raise RuntimeError(
            f"{method} was finalized with dynamic policy={fdr_meta['fdr_strategy']!r}; "
            f"the production comparison requires {UNIFIED_DYNAMIC_POLICY!r}. "
            "Re-finalize existing discovery chunks with "
            "run_stage1_refinalize_dynamic_stagewise.sh; no QTL scan rerun is required."
        )

    d = _read(tqtl_file, need_interaction=False)
    r = _read_replication(rep_file)
    m = _merge(d, r)
    mok = np.isfinite(pd.to_numeric(m.Pvalue_rep, errors="coerce"))
    m = m.loc[mok].copy()
    temporal_rep = pd.to_numeric(m.Pvalue_rep, errors="coerce") < 0.05
    mhit = m.loc[temporal_rep].copy()

    out = {
        "method": method,
        "method_family": "revised_current_v89" if method in REVISED_FOLLOWUP_METHODS else "manuscript_exact",
        "result_label": label,
        "data_mode": data_mode,
        "replication_cohort": "PD",
        "replication_design": "PDBP_only_followup",
        "uses_covariate_adjustment": bool(
            method in {"submitted_followup_mptga", "submitted_followup_ctar_mptga", "lin2018_regression_cov"}
        ),
        "uses_cell_composition_adjustment": bool(
            method in {"submitted_followup_mptga", "submitted_followup_ctar_mptga", "lin2018_regression_cov"}
        ),
        **fdr_meta,
        "n_tqtl_discovery": int(len(d)),
        "n_tqtl_genes_discovery": int(d.Ensembl_ID.astype(str).nunique()),
        "n_tqtl_variants_discovery": int(d.Variant_ID.astype(str).nunique()),
        "n_tqtl_matched": int(len(m)),
        "n_tqtl_genes_matched": int(m.Ensembl_ID.astype(str).nunique()),
        "tqtl_match_fraction": float(len(m) / len(d)) if len(d) else np.nan,
        "n_tqtl_rep_nominal": int(temporal_rep.sum()),
        "n_tqtl_genes_rep_nominal": int(mhit.Ensembl_ID.astype(str).nunique()),
        "n_tqtl_variants_rep_nominal": int(mhit.Variant_ID.astype(str).nunique()),
        "tqtl_rep_nominal_rate": float(temporal_rep.mean()) if len(m) else np.nan,
        "pi1_tqtl": _safe_pi1(m.Pvalue_rep),
        "tqtl_logp_spearman": _spearman_logp(m),
    }

    # Dynamic replication: discovery classification is taken directly from the
    # finalized Stage-I significant_dynamic.parquet, so each method keeps its
    # own recorded interaction-FDR policy. PDBP remains nominal validation.
    dyn_file = pp / "significant_dynamic.parquet"
    out.update({
        "n_dynamic_discovery": np.nan,
        "n_dynamic_genes_discovery": np.nan,
        "n_dynamic_matched": np.nan,
        "n_dynamic_temporal_rep": np.nan,
        "n_dynamic_rep_nominal": np.nan,
        "n_dynamic_genes_rep_nominal": np.nan,
        "dynamic_rep_nominal_rate": np.nan,
        "dynamic_rep_rate_given_temporal_rep": np.nan,
        "pi1_dynamic_interaction": np.nan,
        "dynamic_interaction_logp_spearman": np.nan,
    })
    if dyn_file.exists() and "Interaction_Pvalue" in r.columns:
        dd = _read(dyn_file, need_interaction=True)
        mm = _merge(dd, r)
        ok = (
            np.isfinite(pd.to_numeric(mm.Pvalue_rep, errors="coerce")) &
            np.isfinite(pd.to_numeric(mm.Interaction_Pvalue_rep, errors="coerce"))
        )
        mm = mm.loc[ok].copy()
        temp_ok = pd.to_numeric(mm.Pvalue_rep, errors="coerce") < 0.05
        int_ok = pd.to_numeric(mm.Interaction_Pvalue_rep, errors="coerce") < 0.05
        hit = temp_ok & int_ok
        out.update({
            "n_dynamic_discovery": int(len(dd)),
            "n_dynamic_genes_discovery": int(dd.Ensembl_ID.astype(str).nunique()),
            "n_dynamic_matched": int(len(mm)),
            "n_dynamic_temporal_rep": int(temp_ok.sum()),
            "n_dynamic_rep_nominal": int(hit.sum()),
            "n_dynamic_genes_rep_nominal": int(mm.loc[hit, "Ensembl_ID"].astype(str).nunique()),
            "dynamic_rep_nominal_rate": float(hit.mean()) if len(mm) else np.nan,
            "dynamic_rep_rate_given_temporal_rep": float(hit.sum() / temp_ok.sum()) if temp_ok.sum() else np.nan,
            "pi1_dynamic_interaction": _safe_pi1(mm.Interaction_Pvalue_rep),
            "dynamic_interaction_logp_spearman": _spearman_logp(
                mm, "Interaction_Pvalue_dis", "Interaction_Pvalue_rep"
            ),
        })

    # Additive-pattern replication is descriptive: discovery additive candidate
    # plus PDBP temporal P<0.05 and PDBP interaction P>0.05.
    add_file = pp / "significant_additive.parquet"
    out.update({
        "n_additive_discovery": np.nan,
        "n_additive_genes_discovery": np.nan,
        "n_additive_matched": np.nan,
        "n_additive_rep_nominal": np.nan,
        "n_additive_genes_rep_nominal": np.nan,
        "additive_rep_nominal_rate": np.nan,
    })
    if add_file.exists() and "Interaction_Pvalue" in r.columns:
        da = _read(add_file, need_interaction=True)
        ma = _merge(da, r)
        ok = (
            np.isfinite(pd.to_numeric(ma.Pvalue_rep, errors="coerce")) &
            np.isfinite(pd.to_numeric(ma.Interaction_Pvalue_rep, errors="coerce"))
        )
        ma = ma.loc[ok].copy()
        hit = (
            (pd.to_numeric(ma.Pvalue_rep, errors="coerce") < 0.05) &
            (pd.to_numeric(ma.Interaction_Pvalue_rep, errors="coerce") > 0.05)
        )
        out.update({
            "n_additive_discovery": int(len(da)),
            "n_additive_genes_discovery": int(da.Ensembl_ID.astype(str).nunique()),
            "n_additive_matched": int(len(ma)),
            "n_additive_rep_nominal": int(hit.sum()),
            "n_additive_genes_rep_nominal": int(ma.loc[hit, "Ensembl_ID"].astype(str).nunique()),
            "additive_rep_nominal_rate": float(hit.mean()) if len(ma) else np.nan,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--methods", default=",".join(ALL_FOLLOWUP_METHODS),
                    help="Comma-separated methods. Default: all 7 manuscript-exact + 2 revised methods")
    ap.add_argument("--data-mode", choices=["auto", "stage1_manuscript", "current_v89"], default="auto",
                    help="auto chooses stage1_manuscript for historical methods and current_v89 for revised methods")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any requested method lacks discovery or PDBP replication output")
    ap.add_argument("--allow-mixed-dynamic-policy", action="store_true",
                    help="Legacy audit only: allow global_global and stagewise discovery sets in one table. Production should not use this.")
    a = ap.parse_args()

    paths = AmpPdPaths(root=a.root)
    methods = [x.strip() for x in a.methods.split(",") if x.strip()]
    unknown = [m for m in methods if m not in ALL_FOLLOWUP_METHODS]
    if unknown:
        ap.error(f"Unknown follow-up methods: {unknown}")

    rows, missing = [], []
    for method in methods:
        mode = method_data_mode(method) if a.data_mode == "auto" else a.data_mode
        expected = method_data_mode(method)
        if mode != expected:
            raise ValueError(
                f"{method} belongs to data_mode={expected}, not {mode}. Use --data-mode auto for a consolidated summary."
            )
        z = summarize_label(
            paths, method, method, data_mode=mode,
            require_unified_dynamic_policy=(not a.allow_mixed_dynamic_policy),
        )
        if z is None:
            missing.append(method)
        else:
            rows.append(z)

    if a.strict and missing:
        raise FileNotFoundError(f"Missing finalized discovery/replication results for: {missing}")

    out = pd.DataFrame(rows)
    dest = paths.qtl_dir / "replication_summary.csv"
    out.to_csv(dest, index=False)
    if missing:
        print(f"[WARN] methods not yet summarized because output is incomplete: {missing}", file=sys.stderr)
    if len(out):
        show = [c for c in [
            "method", "method_family", "data_mode", "fdr_strategy",
            "n_tqtl_discovery", "n_tqtl_matched", "n_tqtl_rep_nominal",
            "tqtl_rep_nominal_rate", "pi1_tqtl", "tqtl_logp_spearman",
            "n_dynamic_discovery", "n_dynamic_temporal_rep", "n_dynamic_rep_nominal",
            "dynamic_rep_nominal_rate", "pi1_dynamic_interaction",
            "n_additive_discovery", "n_additive_rep_nominal", "additive_rep_nominal_rate",
        ] if c in out.columns]
        print(out[show].to_string(index=False))
    print(dest)


if __name__ == "__main__":
    main()
