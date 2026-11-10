from __future__ import annotations

import time
import pandas as pd

from .fixed_mptga import CompleteGridMPTGA


LIN2018_IMPLEMENTATION_TAG = "lin2018_complete_grid_noimpute_matched_v2"


class Lin2018MPTGA:
    """Matched complete-grid Lin2018 MPTGA benchmark.

    This benchmark intentionally uses the same complete scheduled-visit eligibility
    as the submitted follow-up MPTGA and performs *no missing-visit imputation*.
    The Lin benchmark itself remains unadjusted for covariates.  This isolates the
    method/parameterization comparison from sample-selection and imputation effects.

    The statistical model is the complete-grid cubic MPTGA likelihood with discrete
    visit-order AR(1) residual covariance.  The implementation delegates to the
    validated complete-grid engine and then annotates the result with a distinct
    implementation fingerprint so stale imputation-era chunks cannot be reused.
    """

    def __init__(
        self,
        genotype_mode="dummies",
        rho_points=40,
        batch_size=2048,
        device=None,
        dtype="float64",
        rho_grid_mode="linspace_098",
    ):
        self.engine = CompleteGridMPTGA(
            basis_kind="cubic",
            genotype_mode=genotype_mode,
            rho_points=rho_points,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            rho_grid_mode=rho_grid_mode,
        )

    def scan(
        self,
        cov,
        expression,
        genotype,
        cis_map,
        id_col="participant_id",
        visit_col="visit",
        time_col="followup_time",
        required_visits=None,
    ):
        t0 = time.time()
        z = self.engine.scan(
            cov,
            expression,
            genotype,
            cis_map,
            covariate_cols=[],
            id_col=id_col,
            visit_col=visit_col,
            time_col=time_col,
            required_visits=required_visits,
        )
        if z is None or len(z) == 0:
            return pd.DataFrame()
        z = z.copy()
        z["method"] = "lin2018_mptga"
        z["implementation_tag"] = LIN2018_IMPLEMENTATION_TAG
        z["missing_strategy"] = "complete_grid_no_imputation"
        z["sample_eligibility"] = "matched_complete_scheduled_grid"
        z["covariate_adjustment"] = "none_method_default"
        z["runtime_sec_total"] = float(time.time() - t0)
        return z
