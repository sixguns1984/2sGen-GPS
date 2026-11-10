from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np
from scipy.interpolate import BSpline

@dataclass
class TimeBasis:
    kind: str = "cubic"
    df: int = 5
    degree: int = 3
    center: Optional[float] = None
    scale: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    knots: Optional[List[float]] = None

    def fit(self, x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("No finite time values.")
        self.center = float(np.median(x)) if self.center is None else float(self.center)
        s = float(np.std(x, ddof=1))
        self.scale = (s if s > 0 else 1.0) if self.scale is None else float(self.scale)
        self.lower, self.upper = float(np.min(x)), float(np.max(x))
        if self.kind == "spline":
            n_internal = max(0, int(self.df) - self.degree - 1)
            if n_internal:
                q = np.linspace(0, 1, n_internal + 2)[1:-1]
                self.knots = np.quantile(x, q).astype(float).tolist()
            else:
                self.knots = []
        elif self.kind not in {"cubic", "linear"}:
            raise ValueError("kind must be linear, cubic or spline.")
        return self

    @property
    def n_basis(self):
        if self.kind == "linear":
            return 2
        if self.kind == "cubic":
            return 4
        return len(self.knots or []) + self.degree + 1

    def transform(self, x):
        x = np.asarray(x, float)
        if self.center is None:
            raise RuntimeError("Fit the basis first.")
        z = (x - self.center) / self.scale
        if self.kind == "linear":
            return np.column_stack([np.ones_like(z), z])
        if self.kind == "cubic":
            return np.column_stack([np.ones_like(z), z, z*z, z*z*z])
        lo = (self.lower-self.center)/self.scale
        hi = (self.upper-self.center)/self.scale
        ik = [((v-self.center)/self.scale) for v in (self.knots or [])]
        t = np.r_[[lo]*(self.degree+1), ik, [hi]*(self.degree+1)]
        n = len(t)-self.degree-1
        out = np.empty((len(x), n), float)
        for j in range(n):
            c = np.zeros(n); c[j] = 1
            out[:, j] = BSpline(t, c, self.degree, extrapolate=True)(z)
        return out

    def support_mask(self, x):
        x = np.asarray(x, float)

        # Backward compatibility with the historical Stage-I MPTGA
        # raw follow-up-time polynomial:
        #     [1, t, t^2, t^3]
        #
        # The historical basis_json did not store lower/upper bounds,
        # because this basis was not defined through fitted spline
        # boundaries. Therefore every finite follow-up time is within
        # the evaluable support of the legacy polynomial predictor.
        if getattr(self, "parameterization", None) == "raw_followup_time_polynomial":
            return np.isfinite(x)

        # Modern fitted TimeBasis objects should always carry explicit
        # training-support bounds. Fail clearly rather than silently
        # inventing them.
        if self.lower is None or self.upper is None:
            raise RuntimeError(
                "TimeBasis support bounds are unavailable. "
                "The basis must be fitted or reconstructed from complete "
                "serialized metadata before support_mask() is called."
            )

        return np.isfinite(x) & (x >= self.lower) & (x <= self.upper)

    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        """
        Reconstruct TimeBasis from serialized metadata, including
        backward compatibility with historical raw-time polynomial bases.
    
        Historical Stage-I MPTGA results used:
            [1, t, t^2, t^3]
        with parameterization='raw_followup_time_polynomial'.
    
        For that parameterization, center=0 and scale=1 exactly recover
        the original raw-time basis without refitting or recalibration.
        """
        import inspect

        d = dict(d)

        sig = inspect.signature(cls)
        init_keys = {
            name
            for name, par in sig.parameters.items()
            if par.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        init_kwargs = {
            k: v
            for k, v in d.items()
            if k in init_keys
        }

        obj = cls(**init_kwargs)

        # Restore serialization metadata / fitted state that is not an
        # __init__ argument.
        for k, v in d.items():
            if k in init_kwargs:
                continue
            try:
                setattr(obj, k, v)
            except (AttributeError, TypeError):
                pass

        # -----------------------------------------------------------------
        # Backward compatibility for the original Stage-I MPTGA basis.
        #
        # Old basis_json:
        #   kind = cubic
        #   parameterization = raw_followup_time_polynomial
        #   columns = [1, t, t2, t3]
        #
        # Current transform() uses z=(t-center)/scale. Setting
        # center=0 and scale=1 therefore gives z=t and exactly restores
        # [1, t, t^2, t^3].
        # -----------------------------------------------------------------
        parameterization = d.get("parameterization", None)

        if parameterization == "raw_followup_time_polynomial":
            if obj.kind not in {"linear", "cubic"}:
                raise ValueError(
                    "raw_followup_time_polynomial is only compatible "
                    f"with linear/cubic TimeBasis, got kind={obj.kind!r}"
                )

            obj.center = 0.0
            obj.scale = 1.0

        # For modern serialized bases, center/scale should already have
        # been restored. Do not silently fit a new basis because that
        # would change the meaning of previously estimated coefficients.
        if obj.center is None or obj.scale is None:
            raise ValueError(
                "Serialized TimeBasis does not contain sufficient fitted "
                "state for prediction and is not a recognized historical "
                f"parameterization. keys={sorted(d.keys())}, "
                f"parameterization={parameterization!r}"
            )

        return obj

