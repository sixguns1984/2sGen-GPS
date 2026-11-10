from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize_scalar
from .basis import TimeBasis


def ar1_matrix(m, rho):
    d = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    return np.power(float(rho), d)


def ct_ar_matrix(time, phi):
    t = np.asarray(time, float)
    return np.exp(-float(phi) * np.abs(t[:, None] - t[None, :]))


@dataclass
class SmoothCTCovariance:
    variance_basis: dict
    variance_coef: list[float]
    phi: float
    jitter: float = 1e-6

    def sigma(self, age):
        b = TimeBasis.from_dict(self.variance_basis)
        V = b.transform(np.asarray(age, float))
        lv = V @ np.asarray(self.variance_coef, float)
        return np.sqrt(np.exp(np.clip(lv, -20, 20)))

    def matrix(self, age):
        a = np.asarray(age, float)
        s = self.sigma(a)
        R = ct_ar_matrix(a, self.phi)
        S = s[:, None] * R * s[None, :]
        S.flat[::len(a) + 1] += self.jitter
        return S

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _ridge(A, y, alpha=1e-3):
    P = np.eye(A.shape[1]) * alpha
    P[0, 0] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ y)


def prepare_grouped_time_cache(time, groups):
    """Cache participant indices and pairwise time distances by visit count.

    The chronological-age model has irregular ages but very short per-person
    series.  NumPy can Cholesky-factor a stack of same-sized matrices at once,
    so grouping participants by the number of observed samples removes the
    Python participant loop without changing the likelihood.
    """
    t = np.asarray(time, float)
    by_size = {}
    for ix in groups:
        ix = np.asarray(ix, dtype=int)
        if len(ix) == 0:
            continue
        by_size.setdefault(len(ix), []).append(ix)

    out = {}
    for m, rows in by_size.items():
        idx = np.stack(rows, axis=0)
        tt = t[idx]
        diffs = np.abs(tt[:, :, None] - tt[:, None, :])
        out[int(m)] = {"idx": idx, "time": tt, "diffs": diffs}
    return out


def _batched_logdet_cholesky(L):
    return 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)


def fit_smooth_ct_covariance(age, residual, groups, variance_df=5,
                             variance_ridge=1e-2, phi_bounds=(1e-3, 10),
                             jitter=1e-6, grouped_cache=None):
    age = np.asarray(age, float)
    r = np.asarray(residual, float)
    vb = TimeBasis("spline", df=variance_df).fit(age)
    V = vb.transform(age)
    floor = np.nanmedian(r * r) * 1e-6 + 1e-12
    coef = _ridge(V, np.log(np.maximum(r * r, floor)), variance_ridge)
    sig = np.sqrt(np.exp(np.clip(V @ coef, -20, 20)))
    u = r / np.maximum(sig, 1e-8)

    cache = grouped_cache if grouped_cache is not None else prepare_grouped_time_cache(age, groups)

    def nll(phi):
        val = 0.0
        try:
            for m, item in cache.items():
                idx = item["idx"]
                U = u[idx]
                if m == 1:
                    val += 0.5 * float(np.sum(U * U))
                    continue
                R = np.exp(-float(phi) * item["diffs"])
                diag = np.arange(m)
                R[:, diag, diag] += jitter
                L = np.linalg.cholesky(R)
                z = np.linalg.solve(L, U[..., None])[..., 0]
                val += float(np.sum(0.5 * z * z))
                val += float(np.sum(_batched_logdet_cholesky(L) / 2.0))
        except np.linalg.LinAlgError:
            return np.inf
        return float(val)

    opt = minimize_scalar(nll, bounds=phi_bounds, method="bounded", options={"xatol": 1e-4})
    return SmoothCTCovariance(vb.to_dict(), coef.tolist(), float(opt.x), jitter)


def whiten_grouped(y, matrices, time, groups, covariance, grouped_cache=None):
    """Exact grouped whitening, vectorized over participants of equal series size."""
    y = np.asarray(y, float)
    mats = [np.asarray(x, float) for x in matrices]
    t = np.asarray(time, float)
    yw = np.empty_like(y)
    out = [np.empty_like(x) for x in mats]
    logdet = 0.0

    cache = grouped_cache if grouped_cache is not None else prepare_grouped_time_cache(t, groups)

    # sigma(age) is observation-specific but gene-level covariance parameters are fixed.
    sigma_all = covariance.sigma(t)

    for m, item in cache.items():
        idx = item["idx"]
        if m == 1:
            s = sigma_all[idx]
            # covariance for a singleton is sigma^2 + jitter
            sd = np.sqrt(s * s + covariance.jitter)
            yw[idx] = y[idx] / sd
            for k, X in enumerate(mats):
                out[k][idx] = X[idx] / sd[..., None]
            logdet += float(np.sum(2.0 * np.log(sd)))
            continue

        s = sigma_all[idx]
        R = np.exp(-float(covariance.phi) * item["diffs"])
        S = s[:, :, None] * R * s[:, None, :]
        diag = np.arange(m)
        S[:, diag, diag] += covariance.jitter
        L = np.linalg.cholesky(S)

        Y = y[idx]
        yw[idx] = np.linalg.solve(L, Y[..., None])[..., 0]
        for k, X in enumerate(mats):
            out[k][idx] = np.linalg.solve(L, X[idx])
        logdet += float(np.sum(_batched_logdet_cholesky(L)))

    return yw, out, float(logdet)
