from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.metrics import average_precision_score, roc_auc_score

from ..utils import bh_fdr, storey_pi1




def _stagewise_interaction_adjusted(screen_q, interaction_p, alpha=0.05):
    """Heller/stageR-style one-confirmation stage-wise adjustment.

    Stage I screens all G hypotheses by BH at ``alpha``.  If R pass, the single
    confirmation hypothesis for each selected unit is tested at alpha*R/G.
    Returning p*G/R gives an adjusted confirmation p-value on the original alpha
    scale.  Values outside the screened set remain NaN.
    """
    screen_q = np.asarray(screen_q, float)
    interaction_p = np.asarray(interaction_p, float)
    selected = np.isfinite(screen_q) & (screen_q < alpha) & np.isfinite(interaction_p)
    G = int(np.sum(np.isfinite(screen_q)))
    R = int(np.sum(np.isfinite(screen_q) & (screen_q < alpha)))
    out = np.full(len(screen_q), np.nan, float)
    if G > 0 and R > 0:
        out[selected] = np.minimum(1.0, interaction_p[selected] * G / R)
    return out, G, R, (alpha * R / G if G > 0 else 0.0)

def _rate(mask, denom_mask):
    mask = np.asarray(mask, bool)
    denom_mask = np.asarray(denom_mask, bool)
    denom = int(np.sum(denom_mask))
    return float(np.sum(mask & denom_mask) / denom) if denom else np.nan


def simulation_metrics(result, truth):
    """Stage-I benchmark metrics using the manuscript's hierarchical QTL definitions.

    Main benchmark endpoint for every method is tQTL discovery (overall P/FDR).
    Dynamic/additive metrics are secondary and are emitted only when a method returns
    an explicit Interaction_Pvalue.

    Discovery definitions used here:
      tQTL:        global overall BH-FDR < 0.05
      dynamic:     tQTL + Heller/stageR-style stage-wise interaction confirmation
      additive:    tQTL AND interaction nominal P > 0.05
      intermediate:tQTL AND interaction P <= 0.05 but no stage-wise confirmation

    The confirmation level is alpha_II = 0.05 * R/G, where G is the number of
    screened pair hypotheses and R is the number passing the temporal BH screen.
    For methods with an explicit interaction test, the benchmark additionally reports
    the old global/global rule and naive within-tQTL subset BH as sensitivity analyses.
    """
    m = truth.merge(result, on=["Variant_ID", "Ensembl_ID"], how="inner", validate="one_to_one")
    if not len(m):
        return {}
    p = pd.to_numeric(m.Pvalue, errors="coerce").to_numpy(float)
    finite = np.isfinite(p)
    m = m.loc[finite].copy()
    p = p[finite]
    alt = m.is_alt.astype(bool).to_numpy()
    null = ~alt
    score = -np.log10(np.clip(p, 1e-300, 1.0))
    q = bh_fdr(p)

    d = {
        "n_tested": int(len(m)),
        "n_null": int(null.sum()),
        "n_alt": int(alt.sum()),
        "n_dynamic": int(m.is_dynamic.astype(bool).sum()),
        "n_static_alt": int(m.is_static_alt.astype(bool).sum()),
        "auc": float(roc_auc_score(alt.astype(int), score)) if len(np.unique(alt)) > 1 else np.nan,
        "auprc": float(average_precision_score(alt.astype(int), score)) if len(np.unique(alt)) > 1 else np.nan,
        "pi1_all": float(storey_pi1(p)),
    }
    for thr, tag in [(0.05, "p05"), (0.01, "p01"), (0.001, "p001")]:
        called = p < thr
        d[f"type1_{tag}"] = _rate(called, null)
        d[f"power_{tag}"] = _rate(called, alt)
        d[f"empirical_fdr_{tag}"] = float(np.mean(null[called])) if called.any() else 0.0
        d[f"n_called_{tag}"] = int(called.sum())

    called_t = q < 0.05
    d["power_FDR05"] = _rate(called_t, alt)
    d["type1_FDR05"] = _rate(called_t, null)
    d["empirical_fdr_FDR05"] = float(np.mean(null[called_t])) if called_t.any() else 0.0
    d["n_called_FDR05"] = int(called_t.sum())

    # Manuscript-facing aliases make it explicit that these are temporal-tQTL metrics.
    d["tqtl_power_p05"] = d["power_p05"]
    d["tqtl_type1_p05"] = d["type1_p05"]
    d["tqtl_power_FDR05"] = d["power_FDR05"]
    d["tqtl_type1_FDR05"] = d["type1_FDR05"]
    d["tqtl_empirical_fdr_FDR05"] = d["empirical_fdr_FDR05"]
    if null.any():
        pn = p[null]
        d["null_p_median"] = float(np.median(pn))
        ks = kstest(pn, "uniform")
        d["null_uniform_ks"] = float(ks.statistic)
        d["null_uniform_ks_p"] = float(ks.pvalue)

    # MAF-stratified tQTL power when MAF truth is available.
    if "maf" in m.columns:
        maf = pd.to_numeric(m.maf, errors="coerce").to_numpy(float)
        for lo, hi, tag in [(0.08, 0.15, "008_015"), (0.15, 0.30, "015_030"), (0.30, 0.45, "030_045")]:
            aa = alt & np.isfinite(maf) & (maf >= lo) & (maf < hi if hi < 0.45 else maf <= hi)
            d[f"n_alt_maf_{tag}"] = int(aa.sum())
            d[f"power_p05_maf_{tag}"] = _rate(p < 0.05, aa)
            d[f"power_FDR05_maf_{tag}"] = _rate(called_t, aa)

    if "Interaction_Pvalue" in m.columns:
        ip = pd.to_numeric(m.Interaction_Pvalue, errors="coerce").to_numpy(float)
        ok = np.isfinite(ip)
        dyn = m.is_dynamic.astype(bool).to_numpy() & ok
        sta = m.is_static_alt.astype(bool).to_numpy() & ok
        nul = (~m.is_alt.astype(bool).to_numpy()) & ok
        qd_stagewise, G_screen, R_screen, alpha_confirmation = _stagewise_interaction_adjusted(
            q, ip, alpha=0.05
        )
        d["hierarchical_screen_G"] = int(G_screen)
        d["hierarchical_screen_R"] = int(R_screen)
        d["hierarchical_confirmation_alpha"] = float(alpha_confirmation)

        # Interaction-only null calibration, separated from the overall temporal screen.
        # This is especially useful for the CTAR finite-sample calibration checks.
        for thr, tag in [(0.05, "p05"), (0.01, "p01"), (0.001, "p001")]:
            icall = ok & (ip < thr)
            d[f"interaction_type1_static_{tag}"] = _rate(icall, sta)
            d[f"interaction_type1_null_{tag}"] = _rate(icall, nul)
            d[f"interaction_power_dynamic_{tag}"] = _rate(icall, dyn)

        # Nominal conjunction retained for continuity.
        dyn_call_p05 = (p < 0.05) & (ip < 0.05) & ok
        d["dynamic_power_p05"] = _rate(dyn_call_p05, dyn)
        d["dynamic_type1_static_p05"] = _rate(dyn_call_p05, sta)
        d["dynamic_type1_null_p05"] = _rate(dyn_call_p05, nul)
        d["dynamic_n_called_p05"] = int(dyn_call_p05.sum())
        false_dyn_p05 = (~m.is_dynamic.astype(bool).to_numpy()) & dyn_call_p05
        d["dynamic_empirical_fdr_p05"] = (
            float(false_dyn_p05.sum() / dyn_call_p05.sum()) if dyn_call_p05.any() else 0.0
        )

        # Primary revised definition: global temporal BH screen followed by the
        # stage-wise single-confirmation interaction threshold alpha*R/G.
        dyn_stagewise = called_t & (qd_stagewise < 0.05) & ok
        false_stagewise = (~m.is_dynamic.astype(bool).to_numpy()) & dyn_stagewise
        d["dynamic_power_stagewise_FDR05"] = _rate(dyn_stagewise, dyn)
        d["dynamic_type1_static_stagewise_FDR05"] = _rate(dyn_stagewise, sta)
        d["dynamic_type1_null_stagewise_FDR05"] = _rate(dyn_stagewise, nul)
        d["dynamic_empirical_fdr_stagewise_FDR05"] = (
            float(false_stagewise.sum() / dyn_stagewise.sum()) if dyn_stagewise.any() else 0.0
        )
        d["dynamic_n_called_stagewise_FDR05"] = int(dyn_stagewise.sum())

        # Backward-compatible aliases: these now refer to the revised primary
        # stage-wise procedure, matching the real-data Stage-I finalizer.
        d["dynamic_power_FDR05"] = d["dynamic_power_stagewise_FDR05"]
        d["dynamic_type1_static_FDR05"] = d["dynamic_type1_static_stagewise_FDR05"]
        d["dynamic_type1_null_FDR05"] = d["dynamic_type1_null_stagewise_FDR05"]
        d["dynamic_empirical_fdr_FDR05"] = d["dynamic_empirical_fdr_stagewise_FDR05"]
        d["dynamic_n_called_FDR05"] = d["dynamic_n_called_stagewise_FDR05"]

        # Sensitivity 1: old independent global/global interaction BH.
        qd_global = np.full(len(m), np.nan, float)
        if np.any(ok):
            qd_global[ok] = bh_fdr(ip[ok])
        dyn_global = called_t & (qd_global < 0.05) & ok
        false_global = (~m.is_dynamic.astype(bool).to_numpy()) & dyn_global
        d["dynamic_power_globalglobal_FDR05"] = _rate(dyn_global, dyn)
        d["dynamic_type1_static_globalglobal_FDR05"] = _rate(dyn_global, sta)
        d["dynamic_type1_null_globalglobal_FDR05"] = _rate(dyn_global, nul)
        d["dynamic_empirical_fdr_globalglobal_FDR05"] = (
            float(false_global.sum() / dyn_global.sum()) if dyn_global.any() else 0.0
        )
        d["dynamic_n_called_globalglobal_FDR05"] = int(dyn_global.sum())

        # Sensitivity 2: naive interaction BH only inside the selected temporal tQTLs.
        # This is diagnostic only; simulation quantifies whether its FDR is inflated.
        qd_subset = np.full(len(m), np.nan, float)
        subset = called_t & ok
        if np.any(subset):
            qd_subset[subset] = bh_fdr(ip[subset])
        dyn_subset = subset & (qd_subset < 0.05)
        false_subset = (~m.is_dynamic.astype(bool).to_numpy()) & dyn_subset
        d["dynamic_power_subsetBH_FDR05"] = _rate(dyn_subset, dyn)
        d["dynamic_type1_static_subsetBH_FDR05"] = _rate(dyn_subset, sta)
        d["dynamic_type1_null_subsetBH_FDR05"] = _rate(dyn_subset, nul)
        d["dynamic_empirical_fdr_subsetBH_FDR05"] = (
            float(false_subset.sum() / dyn_subset.sum()) if dyn_subset.any() else 0.0
        )
        d["dynamic_n_called_subsetBH_FDR05"] = int(dyn_subset.sum())

        # Additive/intermediate patterns use the revised primary stage-wise call.
        add_call = called_t & (ip > 0.05) & ok
        d["additive_pattern_power_FDR05"] = _rate(add_call, sta)
        d["additive_pattern_false_dynamic_FDR05"] = _rate(add_call, dyn)
        d["additive_n_called_FDR05"] = int(add_call.sum())

        interm_call = called_t & (ip <= 0.05) & ~(qd_stagewise < 0.05) & ok
        d["intermediate_n_called_FDR05"] = int(interm_call.sum())

    d["runtime_sec"] = float(result.runtime_sec_total.iloc[0]) if "runtime_sec_total" in result else np.nan
    d["median_n_participants"] = float(result.n_participants.median()) if "n_participants" in result else np.nan
    d["median_n_obs"] = float(result.n_obs.median()) if "n_obs" in result else np.nan
    for c in ["rho_true", "confounding_strength", "missing_rate"]:
        if c in m:
            d[c] = float(pd.to_numeric(m[c], errors="coerce").median())
    if "covariance_mode" in m:
        d["covariance_mode"] = str(m.covariance_mode.iloc[0])
    return d


def aggregate_replicates(df, group_cols=("scenario", "method", "covariate_mode")):
    if df.empty:
        return df.copy()
    numeric = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_replicates"] = int(g.replicate.nunique()) if "replicate" in g else int(len(g))
        for c in numeric:
            x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(float)
            if len(x) == 0:
                continue
            mean = float(np.mean(x))
            sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            half = 1.96 * sd / np.sqrt(len(x)) if len(x) > 1 else 0.0
            row[f"{c}_mean"] = mean
            row[f"{c}_sd"] = sd
            row[f"{c}_ci95_low"] = mean - half
            row[f"{c}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def real_replication_metrics(discovery, replication):
    """Manuscript-compatible real-data PPMI->PDBP benchmark metrics.

    Parameters are per-pair result tables (not prefiltered hit tables). Temporal
    discovery uses global BH; dynamic discovery uses the Heller/stageR-style
    stage-wise interaction confirmation stored by the revised Stage-I finalizer
    when available. Dynamic replication requires both replication temporal P<0.05
    and replication interaction P<0.05.
    """
    keys = ["Variant_ID", "Ensembl_ID"]
    d = discovery.copy()
    r = replication.copy()
    if d.duplicated(keys).any() or r.duplicated(keys).any():
        raise ValueError("Discovery/replication results must contain unique SNP-gene pairs")

    p = pd.to_numeric(d["Pvalue"], errors="coerce").to_numpy(float)
    if "FDR" in d.columns:
        tq = pd.to_numeric(d["FDR"], errors="coerce").to_numpy(float) < 0.05
    else:
        q = bh_fdr(p)
        tq = q < 0.05
    dt = d.loc[tq].copy()
    m = dt.merge(r, on=keys, how="inner", suffixes=("_dis", "_rep"), validate="one_to_one")
    prep = pd.to_numeric(m.get("Pvalue_rep"), errors="coerce")
    mok = np.isfinite(prep)
    m = m.loc[mok].copy()
    prep = pd.to_numeric(m["Pvalue_rep"], errors="coerce").to_numpy(float)
    hit = prep < 0.05

    out = {
        "n_tqtl_discovery": int(len(dt)),
        "n_tqtl_matched": int(len(m)),
        "tqtl_match_fraction": float(len(m) / len(dt)) if len(dt) else np.nan,
        "n_tqtl_rep_nominal": int(hit.sum()),
        "tqtl_rep_nominal_rate": float(hit.mean()) if len(m) else np.nan,
        "pi1_tqtl": float(storey_pi1(prep)) if len(m) else np.nan,
    }
    if len(m) >= 3:
        x = -np.log10(np.clip(pd.to_numeric(m["Pvalue_dis"], errors="coerce"), 1e-300, 1))
        y = -np.log10(np.clip(pd.to_numeric(m["Pvalue_rep"], errors="coerce"), 1e-300, 1))
        out["tqtl_logp_spearman"] = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    else:
        out["tqtl_logp_spearman"] = np.nan

    if "Interaction_Pvalue" in d.columns and "Interaction_Pvalue" in r.columns:
        d2 = d.copy()
        if "Interaction_FDR" in d2.columns:
            iq_all = pd.to_numeric(d2["Interaction_FDR"], errors="coerce").to_numpy(float)
        else:
            ip_all = pd.to_numeric(d2["Interaction_Pvalue"], errors="coerce").to_numpy(float)
            screen_q = (
                pd.to_numeric(d2["FDR"], errors="coerce").to_numpy(float)
                if "FDR" in d2.columns else bh_fdr(pd.to_numeric(d2["Pvalue"], errors="coerce").to_numpy(float))
            )
            iq_all, _, _, _ = _stagewise_interaction_adjusted(screen_q, ip_all, alpha=0.05)
        d2["_Interaction_FDR_stagewise"] = iq_all
        dt2 = d2.loc[tq].copy()
        ip = pd.to_numeric(dt2["Interaction_Pvalue"], errors="coerce").to_numpy(float)
        iq = pd.to_numeric(dt2["_Interaction_FDR_stagewise"], errors="coerce").to_numpy(float)
        ok = np.isfinite(ip)
        dyn = dt2.loc[ok & (iq < 0.05)].copy()
        add = dt2.loc[ok & (ip > 0.05)].copy()

        md = dyn.merge(r, on=keys, how="inner", suffixes=("_dis", "_rep"), validate="one_to_one")
        ipr = pd.to_numeric(md.get("Interaction_Pvalue_rep"), errors="coerce")
        dok = np.isfinite(ipr)
        md = md.loc[dok].copy()
        dhit = (
            pd.to_numeric(md["Pvalue_rep"], errors="coerce").to_numpy(float) < 0.05
        ) & (
            pd.to_numeric(md["Interaction_Pvalue_rep"], errors="coerce").to_numpy(float) < 0.05
        )
        dtemp = pd.to_numeric(md["Pvalue_rep"], errors="coerce").to_numpy(float) < 0.05
        out.update({
            "n_dynamic_discovery": int(len(dyn)),
            "n_dynamic_matched": int(len(md)),
            "n_dynamic_temporal_rep": int(dtemp.sum()),
            "n_dynamic_rep_nominal": int(dhit.sum()),
            "dynamic_rep_nominal_rate": float(dhit.mean()) if len(md) else np.nan,
            "dynamic_rep_rate_given_temporal_rep": float(dhit.sum() / dtemp.sum()) if dtemp.sum() else np.nan,
            "pi1_dynamic_interaction": float(storey_pi1(md["Interaction_Pvalue_rep"])) if len(md) else np.nan,
        })

        ma = add.merge(r, on=keys, how="inner", suffixes=("_dis", "_rep"), validate="one_to_one")
        ar = pd.to_numeric(ma.get("Pvalue_rep"), errors="coerce")
        ai = pd.to_numeric(ma.get("Interaction_Pvalue_rep"), errors="coerce")
        aok = np.isfinite(ar) & np.isfinite(ai)
        ma = ma.loc[aok].copy()
        ahit = (
            pd.to_numeric(ma["Pvalue_rep"], errors="coerce").to_numpy(float) < 0.05
        ) & (
            pd.to_numeric(ma["Interaction_Pvalue_rep"], errors="coerce").to_numpy(float) > 0.05
        )
        out.update({
            "n_additive_discovery": int(len(add)),
            "n_additive_matched": int(len(ma)),
            "n_additive_rep_nominal": int(ahit.sum()),
            "additive_rep_nominal_rate": float(ahit.mean()) if len(ma) else np.nan,
        })

    return out, m
