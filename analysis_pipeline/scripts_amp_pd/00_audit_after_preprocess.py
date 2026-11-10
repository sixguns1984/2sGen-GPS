#!/usr/bin/env python
from pathlib import Path
import argparse, sys, json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, LEGACY_TIMES_BY_COHORT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    a = p.parse_args()
    paths = AmpPdPaths(root=a.root)
    meta = pd.read_parquet(paths.data_dir / "metadata" / "amp_pd_samples.parquet")
    rows = []
    for cohort in ["BF", "PD", "PP", "REP"]:
        if cohort == "REP":
            m = meta[meta.cohort.astype(str).isin(["PD", "BF"])].copy()
        else:
            m = meta[meta.cohort.astype(str) == cohort].copy()
        if m.empty:
            continue
        print(f"\n### {cohort}")
        print("samples", len(m), "participants", m.participant_id.nunique(),
              "age", float(m.age.min()), float(m.age.max()))
        tab = m.groupby("followup_time").agg(samples=("sample_id", "nunique"),
                                             participants=("participant_id", "nunique"))
        print(tab)
        if cohort in LEGACY_TIMES_BY_COHORT:
            times = LEGACY_TIMES_BY_COHORT[cohort]
            complete = 0
            for pid, g in m.groupby("participant_id"):
                vals = set(g.followup_time.astype(float))
                if all(t in vals for t in times):
                    complete += 1
            print("complete participants for cohort-specific legacy times", times, ":", complete)
        else:
            print("no legacy fixed-grid analysis for this cohort")

        pc_cols = [c for c in m.columns if c.startswith(("gPC_", "ePC_"))]
        rows.append({
            "cohort": cohort,
            "samples": len(m),
            "participants": m.participant_id.nunique(),
            "age_min": float(m.age.min()),
            "age_max": float(m.age.max()),
            "n_genotype_pc_cols_nonmissing": int(sum(m[c].notna().any() for c in pc_cols if c.startswith("gPC_"))),
            "n_expression_pc_cols_nonmissing": int(sum(m[c].notna().any() for c in pc_cols if c.startswith("ePC_"))),
        })
    out = paths.data_dir / "metadata" / "cohort_visit_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nWrote", out)
    unmapped = paths.data_dir / "metadata" / "unmapped_longitudinal_visits.csv"
    if unmapped.exists():
        print("\nUnmapped longitudinal visits:")
        print(pd.read_csv(unmapped).to_string(index=False))

    be_report = paths.data_dir / "metadata" / "expression_pc_BE_selection.json"
    if be_report.exists():
        print("\nExpression PC selection by PCAForQTL BE:")
        import json
        z = json.loads(be_report.read_text())
        for group, info in z.items():
            n_genes = info.get("n_genes_used", info.get("n_genes_input", info.get("n_genes")))
            n_donors = info.get("n_donors_BE")
            print(group, "K_BE =", info.get("K_BE"), "K_elbow =", info.get("K_elbow"),
                  "samples =", info.get("n_samples"), "donors_BE =", n_donors,
                  "genes =", n_genes)
    else:
        print("\nExpression PC BE selection has not been run yet.")

if __name__ == "__main__":
    main()
