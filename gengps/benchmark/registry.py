from __future__ import annotations

from ..methods.fixed_mptga import CompleteGridMPTGA
from ..methods.lin2018 import Lin2018MPTGA
from ..methods.comparators import (
    PolynomialRegressionScanner,
    UnionFisherScanner,
    AR1AssociationScanner,
    MANOVAScanner,
)
from ..methods.age_tqtl import AgeTQTLScanner
from ..methods.ctar_mptga import ContinuousTimeMPTGA


FOLLOWUP_BENCHMARK_METHODS = [
    "submitted_followup_mptga",
    "submitted_followup_ctar_mptga",
    "lin2018_mptga",
    "lin2018_regression",
    "lin2018_regression_cov",
    "lin2018_ar",
    "union",
    "fisher",
    "manova",
]

OPTIONAL_AGE_METHODS = [
    "proposed_age_cubic",
    "proposed_age_spline",
    "age_cubic_iid_cov",
]

# Retained for backward compatibility with optional/exploratory commands.
ALL_METHODS = FOLLOWUP_BENCHMARK_METHODS + OPTIONAL_AGE_METHODS

GPU_METHODS = set(ALL_METHODS) - {"manova"}


def _split_alias(name: str):
    if name.endswith("__nocov"):
        return name[: -len("__nocov")], "nocov"
    return name, None


def build_method(name, cfg):
    name, _ = _split_alias(name)
    dtype = cfg.get("dtype", "float64")
    if name == "submitted_followup_mptga":
        return CompleteGridMPTGA(
            "cubic",
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("rho_points", 40),
            cfg.get("batch_size", 2048),
            cfg.get("device"),
            dtype=dtype,
            rho_grid_mode=cfg.get("rho_grid_mode", "linspace_098"),
        )
    if name == "submitted_followup_ctar_mptga":
        return ContinuousTimeMPTGA(
            "cubic",
            cfg.get("legacy_genotype_mode", "binary"),
            cfg.get("rho_points", 100),
            cfg.get("batch_size", 2048),
            cfg.get("device"),
            dtype=dtype,
            reference_interval=cfg.get("ctar_reference_interval", 1.0),
            rho_max=cfg.get("ctar_rho_max", 0.99),
        )
    if name == "lin2018_mptga":
        return Lin2018MPTGA(
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("rho_points", 40),
            cfg.get("batch_size", 2048),
            cfg.get("device"),
            dtype=dtype,
            rho_grid_mode=cfg.get("rho_grid_mode", "linspace_098"),
        )
    if name == "lin2018_regression":
        return PolynomialRegressionScanner(
            "cubic",
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
            covariates=False,
        )
    if name == "lin2018_regression_cov":
        return PolynomialRegressionScanner(
            "cubic",
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
            covariates=True,
        )
    if name == "lin2018_ar":
        return AR1AssociationScanner(
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
        )
    if name in {"union", "fisher"}:
        return UnionFisherScanner(
            name,
            cfg.get("legacy_genotype_mode", "dummies"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
        )
    if name == "manova":
        return MANOVAScanner(cfg.get("legacy_genotype_mode", "dummies"))
    if name == "proposed_age_cubic":
        return AgeTQTLScanner(
            "cubic",
            5,
            cfg.get("variance_df", 5),
            cfg.get("genotype_mode", "dosage"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
        )
    if name == "proposed_age_spline":
        return AgeTQTLScanner(
            "spline",
            cfg.get("spline_df", 5),
            cfg.get("variance_df", 5),
            cfg.get("genotype_mode", "dosage"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
        )
    if name == "age_cubic_iid_cov":
        return PolynomialRegressionScanner(
            "cubic",
            cfg.get("genotype_mode", "dosage"),
            cfg.get("batch_size", 4096),
            cfg.get("device"),
            dtype=dtype,
            covariates=True,
        )
    raise KeyError(name)


def run_method(name, cov, expr, geno, cis, cfg):
    base_name, alias = _split_alias(name)
    m = build_method(base_name, cfg)
    covars = list(cfg.get("covariates", []))
    if alias == "nocov":
        covars = []
    common = {"id_col": cfg.get("id_col", "participant_id")}

    if base_name == "submitted_followup_mptga":
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            covars,
            visit_col=cfg.get("submitted_visit_col", "followup_time"),
            time_col=cfg.get("followup_col", "followup_time"),
            required_visits=cfg.get("required_visits"),
        )
    elif base_name == "submitted_followup_ctar_mptga":
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            covars,
            id_col=common["id_col"],
            time_col=cfg.get("followup_col", "followup_time"),
        )
    elif base_name == "lin2018_mptga":
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            visit_col=cfg.get("submitted_visit_col", "followup_time"),
            time_col=cfg.get("followup_col", "followup_time"),
            required_visits=cfg.get("required_visits"),
            **common,
        )
    elif base_name in {"lin2018_regression", "lin2018_regression_cov"}:
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            covariate_cols=(covars if base_name == "lin2018_regression_cov" else []),
            id_col=common["id_col"],
            time_col=cfg.get("followup_col", "followup_time"),
        )
    elif base_name == "lin2018_ar":
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            id_col=common["id_col"],
            time_col=cfg.get("followup_col", "followup_time"),
        )
    elif base_name in {"union", "fisher", "manova"}:
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            id_col=common["id_col"],
            visit_col=cfg.get("visit_col", "visit"),
        )
    elif base_name in {"proposed_age_cubic", "proposed_age_spline"}:
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            covars,
            id_col=common["id_col"],
            age_col=cfg.get("age_col", "age"),
        )
    elif base_name == "age_cubic_iid_cov":
        z = m.scan(
            cov,
            expr,
            geno,
            cis,
            covars,
            id_col=common["id_col"],
            time_col=cfg.get("age_col", "age"),
        )
    else:
        raise KeyError(base_name)

    if alias is not None and z is not None and len(z):
        z = z.copy()
        z["benchmark_alias"] = name
        z["covariate_mode"] = alias
    return z
