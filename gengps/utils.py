from __future__ import annotations
import numpy as np
import pandas as pd

def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if not len(pv): return q
    o = np.argsort(pv)
    r = pv[o]
    qr = r * len(r) / np.arange(1, len(r)+1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    tmp = np.empty_like(qr); tmp[o] = np.minimum(qr, 1)
    q[ok] = tmp
    return q

def stagewise_dynamic_fdr(screen_p, interaction_p, alpha=0.05):
    """Unified CTAR-style two-stage dynamic-QTL adjustment.

    Stage I applies BH to the overall/temporal p-values across all finite tests.
    If G tests are finite and R pass the temporal BH screen at ``alpha``, the
    single interaction-confirmation hypothesis for each screened pair is tested
    at ``alpha_II = alpha * R / G``.  The returned interaction-adjusted values
    are ``p_interaction * G / R`` (capped at 1) within the screened set and NaN
    elsewhere.

    This helper is used only for in-memory/diagnostic scanner outputs.  Production
    AMP-PD discovery still recomputes the same rule globally across all chunk
    Parquets in ``gengps.amp_pd.fdr.finalize_method_fdr``.
    """
    screen_p = np.asarray(screen_p, float)
    interaction_p = np.asarray(interaction_p, float)
    if screen_p.shape != interaction_p.shape:
        raise ValueError("screen_p and interaction_p must have the same shape")
    q = bh_fdr(screen_p)
    finite_screen = np.isfinite(screen_p)
    G = int(finite_screen.sum())
    screened = np.isfinite(q) & (q < float(alpha))
    R = int(screened.sum())
    out = np.full_like(interaction_p, np.nan, dtype=float)
    eligible = screened & np.isfinite(interaction_p)
    if G > 0 and R > 0:
        out[eligible] = np.minimum(1.0, interaction_p[eligible] * G / R)
    elif R == 0:
        out[eligible] = 1.0
    return q, out, {
        "G": G,
        "R": R,
        "alpha": float(alpha),
        "alpha_II": (float(alpha) * R / G if G > 0 else 0.0),
        "G_over_R": (float(G / R) if R > 0 else None),
    }

def storey_pi1(p, lambdas=None):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if not len(p): return np.nan
    lambdas = np.arange(.05,.96,.05) if lambdas is None else np.asarray(lambdas)
    pi0 = np.array([(p>l).mean()/(1-l) for l in lambdas])
    pi0hat = float(np.clip(np.median(pi0[lambdas>=.5]), 0, 1))
    return 1-pi0hat

def genotype_channels(g, mode="dosage"):
    """Encode hard-call genotypes for the QTL scanners.

    ``binary`` reproduces the legacy ``category_encoders.BinaryEncoder`` mapping
    used by the original 2sGen-GPS Stage-I code after fitting on [1, 2, -1]:

      0  -> (0, 0)   # reference / unseen category
      1  -> (0, 1)
      2  -> (1, 0)
     -1  -> (1, 1)   # legacy missing hard-call code

    This is intentionally *not* mean imputation.  With an intercept, the two
    binary channels span the same three-class genotype design as the original
    implementation while preserving its explicit -1 -> 11 behavior.
    """
    g = np.asarray(g, float)
    if mode == "dosage":
        return g[..., None]
    if mode == "dummies":
        return np.stack([(g == 1).astype(float), (g == 2).astype(float)], axis=-1)
    if mode == "binary":
        c0 = ((g == 2) | (g == -1)).astype(float)
        c1 = ((g == 1) | (g == -1)).astype(float)
        bad = np.isfinite(g) & ~np.isin(g, [-1.0, 0.0, 1.0, 2.0])
        if bad.any():
            vals = np.unique(g[bad])[:10]
            raise ValueError(f"binary genotype mode received non-hard-call values: {vals.tolist()}")
        # A raw NaN is never silently encoded.  Stage-I raw-WGS readers convert
        # missing calls to -1 before reaching this function.
        if np.isnan(g).any():
            raise ValueError("binary genotype mode received NaN; convert missing hard calls to -1 first")
        return np.stack([c0, c1], axis=-1)
    raise ValueError("genotype_mode must be dosage, dummies, or binary")

def participant_groups(cov, id_col="participant_id"):
    ids = cov[id_col].astype(str).to_numpy()
    return [np.flatnonzero(ids == pid) for pid in pd.unique(ids)]

def stable_logdet(A):
    sign, val = np.linalg.slogdet(A)
    return val if sign > 0 else np.inf

def require_aligned(cov, expr):
    if len(cov) != len(expr) or not cov.index.equals(expr.index):
        raise ValueError("cov and expression must have exactly the same observation index/order.")

def cis_map_from_positions(snps, genes, dist=1_000_000):
    rows=[]
    snps=snps.copy()
    for _,g in genes.iterrows():
        s=snps[snps["chr"].astype(str)==str(g["chr"])]
        x=s[(s["pos"] >= float(g["start"])-dist) & (s["pos"] <= float(g["end"])+dist)]
        rows.extend((str(v),str(g["geneid"])) for v in x["snpid"])
    return pd.DataFrame(rows,columns=["snpid","geneid"])
