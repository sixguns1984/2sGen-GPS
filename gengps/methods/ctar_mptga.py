from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2

from ..basis import TimeBasis
from ..gpu import get_device, safe_solve, configure_cuda_numerics
from ..utils import bh_fdr, genotype_channels, require_aligned, stagewise_dynamic_fdr


def _profile_ll_torch(rss, nobs: int, logdet_total: float):
    rss = torch.clamp(rss, min=1e-30)
    const = float(nobs * np.log(2.0 * np.pi) + logdet_total + nobs)
    return -0.5 * (const + nobs * torch.log(rss / float(nobs)))


@dataclass(frozen=True)
class _TransitionLayout:
    transition_rows: np.ndarray
    previous_rows: np.ndarray
    delta_time: np.ndarray
    participant_ids: np.ndarray
    visit_counts: np.ndarray


def _prepare_layout(cov: pd.DataFrame, id_col: str, time_col: str) -> _TransitionLayout:
    ids = cov[id_col].astype(str).to_numpy()
    t = pd.to_numeric(cov[time_col], errors="coerce").to_numpy(float)
    if not np.isfinite(t).all():
        raise ValueError("Continuous-time MPTGA requires finite follow-up times for every retained observation")
    if len(ids) == 0:
        raise ValueError("No observations supplied to continuous-time MPTGA")

    same = ids[1:] == ids[:-1]
    tr = np.flatnonzero(same) + 1
    prev = tr - 1
    dt = t[tr] - t[prev]
    if len(dt) and np.any(dt <= 0):
        bad = tr[np.flatnonzero(dt <= 0)[:10]]
        ex = [(str(ids[i]), float(t[i - 1]), float(t[i])) for i in bad]
        raise ValueError(
            "Continuous-time AR(1) requires strictly increasing follow-up time within participant. "
            f"Duplicate/non-increasing times found; examples={ex}"
        )
    _, counts = np.unique(ids, return_counts=True)
    return _TransitionLayout(tr, prev, dt, ids, counts.astype(int))


def _ctar_whiten_numpy(A: np.ndarray, layout: _TransitionLayout, rho_ref: float,
                       reference_interval: float):
    """Innovation whitening for exp(-phi*|dt|) / rho^(|dt|/reference_interval).

    The first observation of each participant is unchanged.  For every later
    observation j in a participant, with gap dt from the previous observation,

        a_j = rho_ref ** (dt / reference_interval)
        z_j = (x_j - a_j*x_{j-1}) / sqrt(1-a_j^2)

    This is exactly the Cholesky/innovation representation of a continuous-time
    AR(1) (Ornstein-Uhlenbeck/exponential-correlation) process.
    """
    A = np.asarray(A, float)
    out = A.copy()
    tr = layout.transition_rows
    if len(tr) == 0:
        return out, 0.0

    if reference_interval <= 0:
        raise ValueError("reference_interval must be >0")
    power = layout.delta_time / float(reference_interval)
    if float(rho_ref) == 0.0:
        a = np.zeros_like(power)
    else:
        a = np.power(float(rho_ref), power)
    one_minus = 1.0 - a * a
    if np.any(one_minus <= 0) or not np.isfinite(one_minus).all():
        raise FloatingPointError("Invalid CT-AR innovation variance; check rho/time gaps")
    den = np.sqrt(one_minus)

    prev = layout.previous_rows
    if A.ndim == 1:
        out[tr] = (A[tr] - a * A[prev]) / den
    elif A.ndim == 2:
        out[tr, :] = (A[tr, :] - a[:, None] * A[prev, :]) / den[:, None]
    else:
        raise ValueError("A must be 1D or 2D")

    # det(R_i) = product_j (1-a_ij^2) for the Markov/exponential correlation.
    logdet = float(np.log(one_minus).sum())
    return out, logdet


class ContinuousTimeMPTGA:
    """Covariate-adjusted follow-up tQTL MPTGA with continuous-time AR(1).

    Mean model
    ----------
      expression = common cubic follow-up curve + covariates
                   + genotype-channel x cubic follow-up curve + error

    Residual correlation
    --------------------
      Corr(e_ij, e_ik) = rho_ref ** (|t_ij-t_ik| / reference_interval)
                        = exp(-phi |t_ij-t_ik|)

    Participants may have different visit counts, missing scheduled visits, and
    irregular follow-up times.  No visit imputation or complete-grid filtering
    is performed.  The residual correlation parameter is profile-likelihood
    selected separately under the null, main-effect-only, and full models.

    With equally spaced visits this model family reduces to ordinary discrete
    AR(1), up to the parameterization of rho by the chosen time unit.
    """

    def __init__(
        self,
        basis_kind: str = "cubic",
        genotype_mode: str = "binary",
        rho_points: int = 100,
        batch_size: int = 2048,
        device=None,
        dtype: str = "float64",
        reference_interval: float = 1.0,
        rho_max: float = 0.99,
    ):
        self.basis_kind = str(basis_kind)
        self.genotype_mode = str(genotype_mode)
        self.rho_points = int(rho_points)
        if self.rho_points < 2:
            raise ValueError("rho_points must be >=2")
        self.rho_max = float(rho_max)
        if not (0 < self.rho_max < 1):
            raise ValueError("rho_max must lie in (0,1)")
        self.rho_grid = (
            np.arange(100, dtype=float) / 100.0
            if self.rho_points == 100 and abs(self.rho_max - 0.99) < 1e-12
            else np.linspace(0.0, self.rho_max, self.rho_points)
        )
        self.batch_size = int(batch_size)
        self.reference_interval = float(reference_interval)
        self.device = get_device(device)
        configure_cuda_numerics(self.device, allow_tf32=False)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32

    def _basis(self, t: np.ndarray):
        if self.basis_kind == "cubic":
            B = np.column_stack([np.ones_like(t), t, t * t, t * t * t])
            basis_json = json.dumps({
                "kind": "cubic",
                "parameterization": "raw_followup_time_polynomial",
                "columns": ["1", "t", "t2", "t3"],
            })
            return B, basis_json, "raw_followup_time_polynomial"
        basis = TimeBasis(self.basis_kind, df=5).fit(t)
        return basis.transform(t), json.dumps(basis.to_dict()), "standardized_time_basis"

    def scan(
        self,
        cov,
        expression,
        genotype,
        cis_map,
        covariate_cols=None,
        id_col="participant_id",
        time_col="followup_time",
    ):
        t0 = time.time()
        require_aligned(cov, expression)
        covariate_cols = list(covariate_cols or [])

        # The loader normally delivers this ordering already, but sort again here
        # so the covariance implementation is self-contained and safe.
        work = cov.copy()
        work["__row_order"] = np.arange(len(work))
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        if work[time_col].isna().any():
            raise ValueError("Continuous-time MPTGA received missing follow-up_time")
        order = work.sort_values([id_col, time_col, "__row_order"], kind="mergesort").index
        cov = cov.loc[order].copy()
        expression = expression.loc[order].copy()
        require_aligned(cov, expression)

        t = pd.to_numeric(cov[time_col], errors="raise").to_numpy(float)
        layout = _prepare_layout(cov, id_col, time_col)
        B, basis_json, basis_parameterization = self._basis(t)
        C = (
            cov[covariate_cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if covariate_cols else np.empty((len(cov), 0), float)
        )
        X0 = np.column_stack([B, C])

        ids = cov[id_col].astype(str).to_numpy()
        pid_order = pd.Index(pd.unique(ids), dtype=str)
        G = genotype.copy()
        G.index = G.index.astype(str)
        Gpart = G.reindex(pid_order)
        if Gpart.isna().all(axis=1).any():
            bad = Gpart.index[Gpart.isna().all(axis=1)].tolist()[:10]
            raise KeyError(f"Genotype matrix missing retained participants; examples={bad}")
        pid_to_row = {p: i for i, p in enumerate(pid_order)}
        obs_to_part = np.fromiter((pid_to_row[x] for x in ids), dtype=int, count=len(ids))

        # Precompute the rho-dependent nuisance/basis transforms once per gene chunk.
        # X0/B do not depend on gene or SNP.
        rho_cache = []
        for rho in self.rho_grid:
            Xw, logdet_x = _ctar_whiten_numpy(X0, layout, rho, self.reference_interval)
            Bw, logdet_b = _ctar_whiten_numpy(B, layout, rho, self.reference_interval)
            if abs(logdet_x - logdet_b) > 1e-8:
                raise RuntimeError("Internal CT-AR logdet inconsistency")
            xt = torch.as_tensor(Xw, dtype=self.dtype, device=self.device)
            bwt = torch.as_tensor(Bw, dtype=self.dtype, device=self.device)
            inv = torch.linalg.pinv(xt.T @ xt)
            rho_cache.append((float(rho), xt, bwt, inv, float(logdet_x)))

        by_gene = cis_map.groupby("geneid", sort=False)["snpid"].apply(list)
        out = []
        nobs = len(cov)
        k = B.shape[1]
        vc = layout.visit_counts
        visit_min = int(vc.min()) if len(vc) else 0
        visit_max = int(vc.max()) if len(vc) else 0
        visit_med = float(np.median(vc)) if len(vc) else np.nan
        n_ge2 = int(np.sum(vc >= 2)) if len(vc) else 0

        for gene, snps in by_gene.items():
            gene = str(gene)
            if gene not in expression.columns:
                continue
            ss_all = [str(s) for s in snps if str(s) in G.columns]
            if not ss_all:
                continue

            y = pd.to_numeric(expression[gene], errors="coerce").to_numpy(float)
            if not np.isfinite(y).all():
                # Expression preprocessing should have removed non-finite values.
                # Do not let gene-specific missingness silently change the longitudinal sample set.
                raise ValueError(f"Non-finite expression values for {gene} in retained CT-AR observations")

            # Gene-specific nuisance-only residuals for every rho.  These are kept
            # on-device and reused by every SNP batch for the gene.
            gprof = []
            best_null = -np.inf
            for rho, xt, bwt, inv, logdet in rho_cache:
                yw, _ = _ctar_whiten_numpy(y, layout, rho, self.reference_interval)
                yt = torch.as_tensor(yw, dtype=self.dtype, device=self.device)
                by0 = inv @ (xt.T @ yt)
                yr = yt - xt @ by0
                rss0 = torch.clamp(yr @ yr, min=1e-30)
                ll0 = _profile_ll_torch(rss0, nobs, logdet)
                ll0f = float(ll0.detach().cpu())
                best_null = max(best_null, ll0f)
                gprof.append((rho, xt, bwt, inv, logdet, yr, rss0))

            Gp = Gpart[ss_all]
            for st in range(0, len(ss_all), self.batch_size):
                ss = ss_all[st: st + self.batch_size]
                gpart = Gp[ss].to_numpy(float).T
                channels_part = genotype_channels(gpart, self.genotype_mode)
                channels_obs = channels_part[:, obs_to_part, :]
                bsz, _, nch = channels_obs.shape
                q = nch * k
                main_idx = torch.as_tensor([c * k for c in range(nch)], dtype=torch.long, device=self.device)

                best_full = torch.full((bsz,), -torch.inf, dtype=self.dtype, device=self.device)
                best_main = torch.full((bsz,), -torch.inf, dtype=self.dtype, device=self.device)
                best_rho = torch.full((bsz,), np.nan, dtype=self.dtype, device=self.device)
                best_beta = torch.full((bsz, q), np.nan, dtype=self.dtype, device=self.device)

                ch = torch.as_tensor(channels_obs, dtype=self.dtype, device=self.device)
                for rho, xt, bwt, inv, logdet, yr, rss0 in gprof:
                    Z = (ch[:, :, :, None] * bwt[None, :, None, :]).reshape(bsz, nobs, q)

                    XtZ = torch.einsum("np,bnk->bpk", xt, Z)
                    cx = torch.einsum("pq,bqk->bpk", inv, XtZ)
                    Zr = Z - torch.einsum("np,bpk->bnk", xt, cx)
                    S = torch.einsum("bnk,bnl->bkl", Zr, Zr)
                    tvec = torch.einsum("bnk,n->bk", Zr, yr)
                    beta = safe_solve(S, tvec.unsqueeze(-1)).squeeze(-1)
                    rssf = (
                        rss0
                        - 2.0 * torch.sum(beta * tvec, dim=1)
                        + torch.einsum("bi,bij,bj->b", beta, S, beta)
                    )
                    rssf = torch.clamp(rssf, min=1e-30)
                    llf = _profile_ll_torch(rssf, nobs, logdet)
                    take = llf > best_full
                    best_full = torch.where(take, llf, best_full)
                    best_rho = torch.where(take, torch.full_like(best_rho, rho), best_rho)
                    best_beta = torch.where(take[:, None], beta, best_beta)

                    Sm = S.index_select(1, main_idx).index_select(2, main_idx)
                    tm = tvec.index_select(1, main_idx)
                    bm = safe_solve(Sm, tm.unsqueeze(-1)).squeeze(-1)
                    rssm = (
                        rss0
                        - 2.0 * torch.sum(bm * tm, dim=1)
                        + torch.einsum("bi,bij,bj->b", bm, Sm, bm)
                    )
                    rssm = torch.clamp(rssm, min=1e-30)
                    llm = _profile_ll_torch(rssm, nobs, logdet)
                    best_main = torch.maximum(best_main, llm)

                best_full_np = best_full.detach().cpu().numpy()
                best_main_np = best_main.detach().cpu().numpy()
                best_rho_np = best_rho.detach().cpu().numpy()
                best_beta_np = best_beta.detach().cpu().numpy()
                lrt = 2.0 * (best_full_np - best_null)
                dlrt = 2.0 * (best_full_np - best_main_np)
                pval = chi2.sf(np.maximum(lrt, 0), nch * k)
                dp = chi2.sf(np.maximum(dlrt, 0), nch * (k - 1))

                for i, snp in enumerate(ss):
                    rho = float(best_rho_np[i])
                    phi = np.inf if rho <= 0 else -np.log(rho) / self.reference_interval
                    row = {
                        "Variant_ID": snp,
                        "Ensembl_ID": gene,
                        "Pvalue": float(pval[i]),
                        "Interaction_Pvalue": float(dp[i]),
                        "LRT": float(lrt[i]),
                        "Dynamic_LRT": float(dlrt[i]),
                        "rho": rho,
                        "rho_reference_interval": float(self.reference_interval),
                        "phi": float(phi),
                        "n_obs": int(nobs),
                        "n_participants": int(len(pid_order)),
                        "n_participants_ge2_visits": n_ge2,
                        "visits_per_participant_min": visit_min,
                        "visits_per_participant_median": visit_med,
                        "visits_per_participant_max": visit_max,
                        "basis_kind": self.basis_kind,
                        "basis_parameterization": basis_parameterization,
                        "time_axis": time_col,
                        "method": "submitted_followup_ctar_mptga",
                        "covariance_model": "continuous_time_ar1_exponential",
                        "missing_visit_strategy": "observed_likelihood_no_imputation",
                        "sample_eligibility": "all_available_numeric_followup",
                        "rho_grid_points": int(len(self.rho_grid)),
                        "rho_grid_max": float(self.rho_grid[-1]),
                        "compute_engine": "ctar_innovations_profile_fwl_v1",
                        "basis_json": basis_json,
                    }
                    for j, v in enumerate(best_beta_np[i]):
                        row[f"beta_g_{j}"] = float(v)
                    out.append(row)

        z = pd.DataFrame(out)
        if len(z):
            # Chunk-level q-values are diagnostic only; production Stage-I strips
            # these and recomputes BH globally across all discovery cis pairs.
            q, iq, _ = stagewise_dynamic_fdr(z.Pvalue, z.Interaction_Pvalue, alpha=0.05)
            z["FDR"] = q
            z["Interaction_FDR"] = iq
            z["runtime_sec_total"] = time.time() - t0
        return z
