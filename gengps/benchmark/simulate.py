from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def _normalize_visit_count_probs(probs, n_visits):
    if probs is None:
        return None
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or len(p) != int(n_visits):
        raise ValueError(
            f"visit_count_probs must have one probability for visit counts 1..{n_visits}; "
            f"got length={len(p) if p.ndim == 1 else 'non-1D'}"
        )
    if np.any(~np.isfinite(p)) or np.any(p < 0) or float(p.sum()) <= 0:
        raise ValueError("visit_count_probs must be finite, non-negative, and sum to >0")
    return p / p.sum()


def simulate_tqtl_panel(
    n_participants=300,
    n_pairs=1000,
    scheduled_times=(0.0, 0.5, 1.0, 2.0, 3.0),
    age_range=(30.0, 88.0),
    missing_rate=0.0,
    irregular=False,
    irregular_jitter_sd=0.08,
    visit_count_probs=None,
    visit_selection="random_subset",
    rho=0.5,
    covariance_mode="discrete_visit",
    heteroscedastic=False,
    confounding_strength=0.0,
    alt_fraction=0.5,
    dynamic_fraction=0.7,
    maf_range=(0.08, 0.45),
    static_effect=0.28,
    dynamic_linear=0.34,
    dynamic_quadratic=0.10,
    seed=1,
):
    """Generate a longitudinal cis-tQTL benchmark panel.

    The simulator is aligned to the Stage-I estimand: true genetic effects are
    functions of *follow-up time*. Chronological age is retained as a realistic
    sample covariate, but no age-QTL is simulated.

    Residual covariance
    -------------------
    ``covariance_mode='discrete_visit'`` uses ``rho^|visit_i-visit_j|`` and
    therefore matches fixed-grid AR(1) methods. ``continuous_followup`` instead
    uses ``rho^|t_i-t_j|`` and is the data-generating model for CTAR scenarios.

    Missing / incomplete follow-up
    ------------------------------
    By default, each post-baseline visit is independently removed with probability
    ``missing_rate``. For a cohort-like visit-count distribution, pass
    ``visit_count_probs`` with probabilities for retaining exactly 1,2,...,T
    observations. Baseline is always retained. With ``visit_selection='random_subset'``
    the remaining visits are sampled from the scheduled post-baseline visits;
    ``'prefix'`` produces monotone dropout. When ``visit_count_probs`` is supplied,
    it takes precedence over ``missing_rate``.

    Confounding
    -----------
    The observed participant-level confounder ``U`` affects both allele probability
    and expression. This gives a direct adjusted-vs-unadjusted benchmark for the
    covariate-aware methods without changing the historical comparator definitions.
    """
    rng = np.random.default_rng(seed)
    scheduled_times = np.asarray(scheduled_times, dtype=float)
    n_visits = len(scheduled_times)
    if n_visits < 2:
        raise ValueError("At least two scheduled visits are required")
    visit_count_probs = _normalize_visit_count_probs(visit_count_probs, n_visits)
    if visit_selection not in {"random_subset", "prefix"}:
        raise ValueError("visit_selection must be random_subset or prefix")

    # Zero padding prevents lexical sorting from changing participant order.
    width = max(6, len(str(n_participants)))
    pid = np.array([f"P{i:0{width}d}" for i in range(n_participants)], dtype=object)
    U = rng.normal(size=n_participants)
    span = float(np.max(scheduled_times) - np.min(scheduled_times))
    base_age = rng.uniform(age_range[0], age_range[1] - max(span, 0.1), n_participants)

    rows = []
    n_visits_retained = np.zeros(n_participants, dtype=int)
    for i, p in enumerate(pid):
        ft = scheduled_times.copy()
        if irregular:
            jitter = rng.normal(0.0, float(irregular_jitter_sd), size=n_visits)
            jitter[0] = 0.0
            ft = np.maximum.accumulate(np.maximum(0.0, ft + jitter))

        if visit_count_probs is None:
            keep = np.ones(n_visits, dtype=bool)
            for v in range(1, n_visits):
                if rng.random() < float(missing_rate):
                    keep[v] = False
        else:
            n_keep = int(rng.choice(np.arange(1, n_visits + 1), p=visit_count_probs))
            keep = np.zeros(n_visits, dtype=bool)
            keep[0] = True
            if n_keep > 1:
                if visit_selection == "prefix":
                    keep[1:n_keep] = True
                else:
                    idx = rng.choice(np.arange(1, n_visits), size=n_keep - 1, replace=False)
                    keep[idx] = True

        n_visits_retained[i] = int(keep.sum())
        for v, t in enumerate(ft):
            if not keep[v]:
                continue
            rows.append((p, int(v), float(t), float(base_age[i] + t), float(U[i])))

    cov = pd.DataFrame(
        rows,
        columns=["participant_id", "visit", "followup_time", "age", "U"],
    )
    pindex = {p: i for i, p in enumerate(pid)}
    obs_i = np.array([pindex[x] for x in cov.participant_id], dtype=int)
    nobs = len(cov)

    # Shared longitudinal nuisance trajectory.
    ft_all = cov.followup_time.to_numpy(float)
    ft_mu = float(np.mean(ft_all))
    ft_sd = float(np.std(ft_all)) or 1.0
    zt = (ft_all - ft_mu) / ft_sd
    common = 0.22 * zt - 0.06 * zt**2

    participant_blocks = []
    for _, idx in cov.groupby("participant_id", sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        participant_blocks.append(ii)

    geno = {}
    expr = {}
    truth = []
    n_alt = int(round(n_pairs * alt_fraction))
    n_dyn = int(round(n_alt * dynamic_fraction))
    effective_missing_rate = 1.0 - (float(nobs) / float(n_participants * n_visits))

    for j in range(n_pairs):
        maf = float(rng.uniform(*maf_range))
        # G <- U structure under confounding.
        base_logit = np.log(maf / (1.0 - maf))
        allele_prob = expit(base_logit + 0.45 * float(confounding_strength) * U)
        allele_prob = np.clip(allele_prob, 0.01, 0.99)
        g = rng.binomial(2, allele_prob).astype(float)

        snp = f"snp{j:07d}"
        gene = f"gene{j:07d}"
        geno[snp] = g

        is_alt = j < n_alt
        is_dynamic = is_alt and j < n_dyn
        go = g[obs_i] - float(np.mean(g))
        genetic = np.zeros(nobs, dtype=float)
        if is_alt:
            effect_curve = np.full(nobs, float(static_effect))
            if is_dynamic:
                effect_curve = (
                    effect_curve
                    + float(dynamic_linear) * zt
                    + float(dynamic_quadratic) * zt**2
                )
            genetic = go * effect_curve

        conf = 0.70 * float(confounding_strength) * U[obs_i]
        if heteroscedastic:
            age = cov.age.to_numpy(float)
            za = (age - np.mean(age)) / (np.std(age) or 1.0)
            sigma = np.exp(0.18 * za)
        else:
            sigma = np.ones(nobs, dtype=float)

        eps = np.zeros(nobs, dtype=float)
        for ii in participant_blocks:
            if covariance_mode == "discrete_visit":
                coord = cov.iloc[ii].visit.to_numpy(float)
            elif covariance_mode == "continuous_followup":
                coord = cov.iloc[ii].followup_time.to_numpy(float)
            else:
                raise ValueError(f"Unknown covariance_mode={covariance_mode}")
            dist = np.abs(coord[:, None] - coord[None, :])
            R = np.power(float(rho), dist)
            S = sigma[ii, None] * R * sigma[ii][None, :]
            S.flat[:: len(ii) + 1] += 1e-8
            eps[ii] = rng.multivariate_normal(np.zeros(len(ii)), S)

        expr[gene] = common + genetic + conf + eps
        truth_class = "dynamic" if is_dynamic else ("static" if is_alt else "null")
        truth.append({
            "Variant_ID": snp,
            "Ensembl_ID": gene,
            "truth_class": truth_class,
            "is_alt": bool(is_alt),
            "is_dynamic": bool(is_dynamic),
            "is_static_alt": bool(is_alt and not is_dynamic),
            "maf": maf,
            "rho_true": float(rho),
            "covariance_mode": covariance_mode,
            "confounding_strength": float(confounding_strength),
            "missing_rate": float(effective_missing_rate),
            "scheduled_missing_rate": float(missing_rate),
            "visit_count_mode": "distribution" if visit_count_probs is not None else "independent_missing",
            "irregular": bool(irregular),
            "heteroscedastic": bool(heteroscedastic),
        })

    gt = pd.DataFrame(geno, index=pd.Index(pid, name="participant_id"))
    ex = pd.DataFrame(expr, index=cov.index)
    cm = pd.DataFrame({
        "snpid": [f"snp{j:07d}" for j in range(n_pairs)],
        "geneid": [f"gene{j:07d}" for j in range(n_pairs)],
    })
    cov.attrs["visit_count_distribution"] = {
        int(k): int(v) for k, v in zip(*np.unique(n_visits_retained, return_counts=True))
    }
    return cov, ex, gt, cm, pd.DataFrame(truth)
