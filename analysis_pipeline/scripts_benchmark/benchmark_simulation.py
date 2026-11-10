#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from _bootstrap import bootstrap
bootstrap()

from gengps.benchmark.simulate import simulate_tqtl_panel
from gengps.benchmark.runner import schedule_commands
from gengps.benchmark.registry import FOLLOWUP_BENCHMARK_METHODS
from gengps.benchmark.summarize import simulation_metrics, aggregate_replicates


# Methods for which the benchmark configuration genuinely passes the observed
# confounder U into the model. Their __nocov aliases are run as an explicit
# adjusted-vs-unadjusted ablation.
COVARIATE_AWARE_METHODS = {
    "submitted_followup_mptga",
    "submitted_followup_ctar_mptga",
    "lin2018_regression_cov",
}

# Dynamic-QTL metrics are secondary. These methods are the default focus in
# the manuscript-facing dynamic figure; the plotting script can be overridden.
DYNAMIC_FOCUS_METHODS = {
    "submitted_followup_mptga",
    "submitted_followup_ctar_mptga",
    "lin2018_regression_cov",
}


def _parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def _rho_tag(rho: float) -> str:
    return f"{rho:.2f}".replace("-", "m").replace(".", "p")


def build_factorial_scenarios(
    rho_values: Iterable[float],
    confounding_strength: float,
    scheduled_times: Iterable[float],
    covariance_mode: str,
    alt_fraction: float,
    dynamic_fraction: float,
    static_effect: float,
    dynamic_linear: float,
    dynamic_quadratic: float,
) -> Dict[str, dict]:
    """Build the primary rho x confounding simulation design.

    The design intentionally uses a complete, equally spaced longitudinal grid.
    This isolates the two questions that the manuscript needs the simulation to
    answer: (i) robustness to residual AR(1) correlation and (ii) the effect of
    covariate adjustment. It does *not* attempt to mimic PPMI/PDBP visit patterns.

    With scheduled_times=(0,1,2,3,4), discrete_visit and continuous_followup
    coordinates are numerically aligned, making the rho comparison fair to both
    fixed-grid AR(1) and continuous-time implementations.
    """
    scheduled_times = [float(x) for x in scheduled_times]
    scenarios: Dict[str, dict] = {}
    for rho in rho_values:
        rho = float(rho)
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must satisfy 0 <= rho < 1; got {rho}")
        for confounded, strength in [
            (False, 0.0),
            (True, float(confounding_strength)),
        ]:
            status = "confounded" if confounded else "unconfounded"
            name = f"rho_{_rho_tag(rho)}__{status}"
            scenarios[name] = dict(
                scheduled_times=scheduled_times,
                irregular=False,
                missing_rate=0.0,
                confounding_strength=strength,
                rho=rho,
                covariance_mode=covariance_mode,
                heteroscedastic=False,
                alt_fraction=float(alt_fraction),
                dynamic_fraction=float(dynamic_fraction),
                static_effect=float(static_effect),
                dynamic_linear=float(dynamic_linear),
                dynamic_quadratic=float(dynamic_quadratic),
                # Metadata-only fields are added outside simulate_tqtl_panel.
                _confounding_status=status,
            )
    return scenarios


def make_config(path, d, batch_size, rho_points, dtype, use_covariate=True):
    covars = '["U"]' if use_covariate else '[]'
    # Historical fixed-grid rho_grid_mode="legacy_exact" is only valid for 100 points.
    # Smaller simulation grids use the generic 0..0.98 linspace for speed.
    rho_grid_mode = "legacy_exact" if int(rho_points) == 100 else "linspace_098"
    txt = f'''[qtl]
cov = "{d / 'cov.parquet'}"
expr = "{d / 'expr.parquet'}"
geno = "{d / 'geno.parquet'}"
cis = "{d / 'cis.parquet'}"
id_col = "participant_id"
visit_col = "visit"
submitted_visit_col = "visit"
followup_col = "followup_time"
age_col = "age"
covariates = {covars}
required_visits = [0, 1, 2, 3, 4]

[method]
batch_size = {batch_size}
rho_points = {rho_points}
ctar_reference_interval = 1.0
ctar_rho_max = 0.99
genotype_mode = "binary"
legacy_genotype_mode = "binary"
rho_grid_mode = "{rho_grid_mode}"
dtype = "{dtype}"
'''
    path.write_text(txt)


def _method_covariate_mode(method: str) -> str:
    if method.endswith("__nocov"):
        return "nocov"
    if method in COVARIATE_AWARE_METHODS:
        return "adjusted"
    return "method_default"


def _strip_internal_metadata(pars: dict) -> dict:
    return {k: v for k, v in pars.items() if not k.startswith("_")}


def main():
    p = argparse.ArgumentParser(
        description=(
            "Factorial longitudinal tQTL simulation: residual AR(1) rho x covariate "
            "confounding. tQTL performance is primary; dynamic-QTL metrics are secondary."
        )
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", default=",".join(FOLLOWUP_BENCHMARK_METHODS))
    p.add_argument("--gpus", default="0,1,2", help="Empty string gives CPU-only mode.")
    p.add_argument("--cpu-parallel", type=int, default=4)
    p.add_argument("--n-participants", type=int, default=300)
    p.add_argument("--n-pairs", type=int, default=1000)
    p.add_argument("--n-replicates", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--rho-points", type=int, default=20)
    p.add_argument(
        "--rho-values",
        default="0,0.3,0.6,0.9",
        help="True residual AR(1) coefficients used by the data-generating model.",
    )
    p.add_argument(
        "--scheduled-times",
        default="0,1,2,3,4",
        help="Equally spaced normalized longitudinal times; default aligns discrete and CTAR correlation coordinates.",
    )
    p.add_argument(
        "--covariance-mode",
        choices=["discrete_visit", "continuous_followup"],
        default="discrete_visit",
    )
    p.add_argument("--confounding-strength", type=float, default=0.90)
    p.add_argument("--alt-fraction", type=float, default=0.50)
    p.add_argument("--dynamic-fraction", type=float, default=0.50)
    p.add_argument("--static-effect", type=float, default=0.28)
    p.add_argument("--dynamic-linear", type=float, default=0.34)
    p.add_argument("--dynamic-quadratic", type=float, default=0.10)
    p.add_argument(
        "--no-nocov-ablation",
        action="store_true",
        help="Do not add __nocov aliases for covariate-aware methods.",
    )
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    if not (0.0 <= a.alt_fraction <= 1.0):
        raise ValueError("--alt-fraction must be in [0,1]")
    if not (0.0 <= a.dynamic_fraction <= 1.0):
        raise ValueError("--dynamic-fraction must be in [0,1]")

    rho_values = _parse_float_list(a.rho_values)
    scheduled_times = _parse_float_list(a.scheduled_times)
    if len(scheduled_times) != 5:
        raise ValueError(
            "The current benchmark methods/config expect five scheduled visits. "
            "Provide exactly five --scheduled-times values."
        )

    scenarios = build_factorial_scenarios(
        rho_values=rho_values,
        confounding_strength=a.confounding_strength,
        scheduled_times=scheduled_times,
        covariance_mode=a.covariance_mode,
        alt_fraction=a.alt_fraction,
        dynamic_fraction=a.dynamic_fraction,
        static_effect=a.static_effect,
        dynamic_linear=a.dynamic_linear,
        dynamic_quadratic=a.dynamic_quadratic,
    )

    root = pathlib.Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    methods0 = [x for x in a.methods.split(",") if x]
    gpus = [x for x in a.gpus.split(",") if x]

    # Store the full design once at the root for reproducibility.
    design_rows = []
    for name, pars in scenarios.items():
        design_rows.append({
            "scenario": name,
            "rho_true": float(pars["rho"]),
            "confounding_status": pars["_confounding_status"],
            "confounding_strength": float(pars["confounding_strength"]),
            "covariance_mode": pars["covariance_mode"],
            "scheduled_times": ",".join(map(str, pars["scheduled_times"])),
            "alt_fraction": float(pars["alt_fraction"]),
            "dynamic_fraction": float(pars["dynamic_fraction"]),
            "static_effect": float(pars["static_effect"]),
            "dynamic_linear": float(pars["dynamic_linear"]),
            "dynamic_quadratic": float(pars["dynamic_quadratic"]),
        })
    pd.DataFrame(design_rows).to_csv(root / "simulation_design.csv", index=False)

    root_meta = {
        "scientific_scope": "longitudinal tQTL benchmark; dynamic-QTL evaluation is secondary",
        "primary_factors": ["residual AR(1) rho", "covariate confounding/adjustment"],
        "rho_values": rho_values,
        "scheduled_times": scheduled_times,
        "covariance_mode": a.covariance_mode,
        "confounding_strength": float(a.confounding_strength),
        "n_participants": int(a.n_participants),
        "n_pairs": int(a.n_pairs),
        "n_replicates": int(a.n_replicates),
        "alt_fraction": float(a.alt_fraction),
        "dynamic_fraction_within_alt": float(a.dynamic_fraction),
        "tqtl_primary_metrics": [
            "power_FDR05",
            "type1_p05",
            "empirical_fdr_FDR05",
            "auc",
            "auprc",
            "null_p_median",
            "null_uniform_ks",
        ],
        "dynamic_secondary_metrics": [
            "dynamic_power_stagewise_FDR05",
            "dynamic_type1_static_stagewise_FDR05",
            "dynamic_type1_null_stagewise_FDR05",
            "dynamic_empirical_fdr_stagewise_FDR05",
        ],
        "covariate_aware_methods": sorted(COVARIATE_AWARE_METHODS),
        "dynamic_focus_methods": sorted(DYNAMIC_FOCUS_METHODS),
        "nocov_ablation_enabled": not a.no_nocov_ablation,
    }
    (root / "simulation_design.json").write_text(json.dumps(root_meta, indent=2))

    rows = []
    total_rows = 0
    for si, (name, pars_with_meta) in enumerate(scenarios.items()):
        pars = _strip_internal_metadata(pars_with_meta)
        confounding_status = pars_with_meta["_confounding_status"]
        scenario_rows = 0

        for rep in range(a.n_replicates):
            seed = int(a.seed + si * 100000 + rep)
            d = root / name / f"rep_{rep:03d}"
            d.mkdir(parents=True, exist_ok=True)

            cov, ex, gt, cm, truth = simulate_tqtl_panel(
                n_participants=a.n_participants,
                n_pairs=a.n_pairs,
                seed=seed,
                **pars,
            )
            cov.to_parquet(d / "cov.parquet")
            ex.to_parquet(d / "expr.parquet")
            gt.to_parquet(d / "geno.parquet")
            cm.to_parquet(d / "cis.parquet")
            truth.to_parquet(d / "truth.parquet")

            visit_counts = cov.groupby("participant_id", sort=False).size()
            sim_meta = {
                "scenario": name,
                "replicate": rep,
                "seed": seed,
                "rho_true": float(pars["rho"]),
                "confounding_status": confounding_status,
                "n_participants_requested": a.n_participants,
                "n_participants_observed": int(cov.participant_id.nunique()),
                "n_observations": int(len(cov)),
                "mean_visits_per_participant": float(visit_counts.mean()),
                "median_visits_per_participant": float(visit_counts.median()),
                "visit_count_distribution": {
                    str(int(k)): int(v)
                    for k, v in visit_counts.value_counts().sort_index().items()
                },
                "n_pairs": a.n_pairs,
                "primary_endpoint": "longitudinal tQTL discovery",
                "primary_tqtl_definition": "global BH-FDR < 0.05 on overall temporal association P values",
                "dynamic_endpoint": "secondary",
                "dynamic_definition": (
                    "temporal tQTL screen followed by Heller/stageR-style one-confirmation "
                    "genotype-by-time test at alpha_II = 0.05 * R/G"
                ),
                **pars,
            }
            (d / "simulation_parameters.json").write_text(json.dumps(sim_meta, indent=2))

            cfg = d / "benchmark.toml"
            make_config(cfg, d, a.batch_size, a.rho_points, a.dtype, True)
            methods = list(methods0)

            # Explicit adjusted-vs-unadjusted ablation at every rho and in both
            # confounding states. This separates the value/cost of adjustment from
            # the residual-correlation benchmark itself.
            if not a.no_nocov_ablation:
                for m in methods0:
                    if m in COVARIATE_AWARE_METHODS:
                        alias = f"{m}__nocov"
                        if alias not in methods:
                            methods.append(alias)

            schedule_commands(
                cfg,
                methods,
                d / "results",
                gpus=gpus,
                cpu_parallel=a.cpu_parallel,
                resume=not a.no_resume,
            )

            for method in methods:
                fp = d / "results" / f"{method}.parquet"
                if not fp.exists():
                    raise FileNotFoundError(fp)
                z = pd.read_parquet(fp)
                met = simulation_metrics(z, truth)
                if not met:
                    raise RuntimeError(
                        f"No evaluable results for {name} replicate={rep} method={method}"
                    )
                met.update({
                    "scenario": name,
                    "replicate": int(rep),
                    "seed": seed,
                    "method": method,
                    "base_method": method[:-len("__nocov")] if method.endswith("__nocov") else method,
                    "covariate_mode": _method_covariate_mode(method),
                    "rho_true": float(pars["rho"]),
                    "confounding_status": confounding_status,
                    "confounding_strength": float(pars["confounding_strength"]),
                    "n_participants_simulated": int(a.n_participants),
                    "n_observations_simulated": int(len(cov)),
                    "mean_visits_simulated": float(visit_counts.mean()),
                    "n_pairs_simulated": int(a.n_pairs),
                })
                if "median_n_participants" in met and pd.notna(met["median_n_participants"]):
                    met["participant_retention_fraction"] = float(
                        met["median_n_participants"] / a.n_participants
                    )
                if "median_n_obs" in met and pd.notna(met["median_n_obs"]):
                    met["observation_retention_fraction"] = float(
                        met["median_n_obs"] / len(cov)
                    )
                rows.append(met)
                scenario_rows += 1
                total_rows += 1

            print(
                f"completed scenario={name} rho={pars['rho']:.2f} "
                f"confounding={confounding_status} replicate={rep} seed={seed}; "
                f"scenario_rows={scenario_rows}; cumulative_rows={total_rows}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(
                root / "simulation_summary_by_replicate.csv", index=False
            )

    byrep = pd.DataFrame(rows)
    if byrep.empty:
        raise RuntimeError("Simulation produced no summary rows")

    # Main cross-method comparison: use the adjusted definition for methods that
    # support covariates and the historical/default definition for other methods.
    primary = byrep[~byrep.method.str.endswith("__nocov")].copy()
    primary_summary = aggregate_replicates(primary)

    byrep.to_csv(root / "simulation_summary_by_replicate.csv", index=False)
    aggregate_replicates(byrep).to_csv(root / "simulation_summary.csv", index=False)
    primary.to_csv(root / "simulation_tqtl_primary_by_replicate.csv", index=False)
    primary_summary.to_csv(root / "simulation_tqtl_primary_summary.csv", index=False)

    # Adjusted vs no-covariate ablation.
    ab_names = COVARIATE_AWARE_METHODS | {
        f"{m}__nocov" for m in COVARIATE_AWARE_METHODS
    }
    ab = byrep[byrep.method.isin(ab_names)].copy()
    ab.to_csv(root / "simulation_covariate_ablation_by_replicate.csv", index=False)
    aggregate_replicates(ab).to_csv(
        root / "simulation_covariate_ablation_summary.csv", index=False
    )

    # Dynamic-QTL secondary table: no claim of a universal cross-method benchmark.
    dyn_metric = "dynamic_power_stagewise_FDR05"
    dyn = primary[
        primary[dyn_metric].notna() if dyn_metric in primary.columns else np.zeros(len(primary), dtype=bool)
    ].copy()
    dyn.to_csv(root / "simulation_dynamic_secondary_by_replicate.csv", index=False)
    aggregate_replicates(dyn).to_csv(
        root / "simulation_dynamic_secondary_summary.csv", index=False
    )

    # Compact manuscript-facing table with the principal endpoints.
    keep = [
        "scenario", "replicate", "method", "base_method", "covariate_mode",
        "rho_true", "confounding_status", "confounding_strength",
        "power_FDR05", "type1_p05", "empirical_fdr_FDR05", "auc", "auprc",
        "null_p_median", "null_uniform_ks", "runtime_sec",
        "dynamic_power_stagewise_FDR05",
        "dynamic_type1_static_stagewise_FDR05",
        "dynamic_type1_null_stagewise_FDR05",
        "dynamic_empirical_fdr_stagewise_FDR05",
    ]
    keep = [c for c in keep if c in byrep.columns]
    byrep[keep].to_csv(root / "simulation_key_metrics_by_replicate.csv", index=False)

    print("\nPrimary tQTL replicate-level results:")
    print(primary)
    print("\nPrimary tQTL aggregated results:")
    print(primary_summary)


if __name__ == "__main__":
    main()
