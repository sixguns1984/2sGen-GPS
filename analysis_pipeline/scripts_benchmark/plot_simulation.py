#!/usr/bin/env python3
"""Publication-quality simulation figures for the longitudinal tQTL benchmark.

Outputs
-------
1. simulation_tqtl_primary.{pdf,svg,png}
   Six-panel main figure: power, nominal type-I error and empirical FDR across rho,
   separately for unconfounded and confounded data.
2. simulation_covariate_ablation.{pdf,svg,png}
   Adjusted vs no-covariate comparison for covariate-aware methods.
3. simulation_dynamic_secondary.{pdf,svg,png}
   Secondary dynamic-QTL evaluation: stage-wise power and false dynamic rates.
4. simulation_key_metrics_summary.csv
   Mean, SD, SEM and 95% CI for plotted endpoints.

The palette is a restrained Nature/ggsci-NPG-style palette and all vector text is
kept editable in PDF/SVG output.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# NPG-like publication palette (colorblind-distinguishable for the primary series).
NATURE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

METHOD_LABELS = {
    "submitted_followup_mptga": "MPTGA",
    "submitted_followup_ctar_mptga": "CTAR-MPTGA",
    "lin2018_mptga": "Lin2018-MPTGA",
    "lin2018_regression": "Lin regression",
    "lin2018_regression_cov": "Lin regression + covariates",
    "lin2018_ar": "AR(1) regression",
    "union": "Union",
    "fisher": "Fisher",
    "manova": "MANOVA",
}

DEFAULT_DYNAMIC_METHODS = [
    "submitted_followup_mptga",
    "submitted_followup_ctar_mptga",
    "lin2018_regression_cov",
]


def _parse_csv_list(text: str | None) -> List[str] | None:
    if text is None or str(text).strip() == "":
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _base_method(method: str) -> str:
    suffix = "__nocov"
    return method[:-len(suffix)] if method.endswith(suffix) else method


def _style_rcparams() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.15,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def _clean_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)
    ax.grid(False)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.24, 1.12, label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _ensure_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    if "base_method" not in z.columns:
        z["base_method"] = z["method"].astype(str).map(_base_method)
    if "rho_true" not in z.columns:
        raise ValueError("Input must contain rho_true. Use simulation_summary_by_replicate.csv from the revised benchmark.")
    z["rho_true"] = pd.to_numeric(z["rho_true"], errors="coerce")
    if "confounding_status" not in z.columns:
        if "confounding_strength" not in z.columns:
            raise ValueError("Input must contain confounding_status or confounding_strength")
        cs = pd.to_numeric(z["confounding_strength"], errors="coerce").fillna(0)
        z["confounding_status"] = np.where(cs > 0, "confounded", "unconfounded")
    if "covariate_mode" not in z.columns:
        z["covariate_mode"] = np.where(
            z["method"].astype(str).str.endswith("__nocov"), "nocov", "method_default"
        )
    return z


def _summary(df: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=[*group_cols, "mean", "sd", "sem", "ci95_low", "ci95_high", "n"])
    x = df[[*group_cols, metric]].copy()
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    x = x.dropna(subset=[metric])
    if x.empty:
        return pd.DataFrame(columns=[*group_cols, "mean", "sd", "sem", "ci95_low", "ci95_high", "n"])
    g = x.groupby(list(group_cols), dropna=False, sort=True)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index().rename(columns={"std": "sd", "count": "n"})
    out["sd"] = out["sd"].fillna(0.0)
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["ci95_low"] = out["mean"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean"] + 1.96 * out["sem"]
    return out


def _method_order(df: pd.DataFrame, requested: List[str] | None = None) -> List[str]:
    present = list(dict.fromkeys(df["base_method"].astype(str).tolist()))
    if requested:
        return [m for m in requested if m in present]
    canonical = list(METHOD_LABELS)
    ordered = [m for m in canonical if m in present]
    ordered.extend([m for m in present if m not in ordered])
    return ordered


def _method_styles(methods: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    return {
        m: (NATURE_PALETTE[i % len(NATURE_PALETTE)], MARKERS[i % len(MARKERS)])
        for i, m in enumerate(methods)
    }


def _plot_series(
    ax: mpl.axes.Axes,
    summary: pd.DataFrame,
    methods: Sequence[str],
    styles: Dict[str, Tuple[str, str]],
    label_mode: str = "method",
    linestyle_by_covariate: bool = False,
) -> None:
    for method in methods:
        gm = summary[summary["base_method"].eq(method)].sort_values("rho_true")
        if gm.empty:
            continue
        color, marker = styles[method]
        if linestyle_by_covariate:
            modes = [m for m in ["adjusted", "nocov", "method_default"] if m in gm["covariate_mode"].unique()]
        else:
            modes = [None]
        for mode in modes:
            h = gm if mode is None else gm[gm["covariate_mode"].eq(mode)]
            if h.empty:
                continue
            x = h["rho_true"].to_numpy(float)
            y = h["mean"].to_numpy(float)
            lo = h["ci95_low"].to_numpy(float)
            hi = h["ci95_high"].to_numpy(float)
            ls = "--" if mode == "nocov" else "-"
            fillstyle = "none" if mode == "nocov" else "full"
            base_label = METHOD_LABELS.get(method, method)
            if label_mode == "method_cov" and mode is not None:
                suffix = "unadjusted" if mode == "nocov" else "adjusted"
                label = f"{base_label} — {suffix}"
            else:
                label = base_label
            ax.plot(
                x, y,
                color=color,
                marker=marker,
                linestyle=ls,
                markerfacecolor=("white" if fillstyle == "none" else color),
                markeredgecolor=color,
                markeredgewidth=0.8,
                label=label,
                zorder=3,
            )
            ax.fill_between(x, lo, hi, color=color, alpha=0.11, linewidth=0, zorder=1)


def _save(fig: mpl.figure.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_tqtl_primary(df: pd.DataFrame, out_dir: Path, methods_requested: List[str] | None) -> pd.DataFrame:
    # Primary comparison excludes explicit __nocov aliases. Historical methods whose
    # definition is unadjusted remain included as their own named comparator.
    z = df[~df["method"].astype(str).str.endswith("__nocov")].copy()
    methods = _method_order(z, methods_requested)
    z = z[z["base_method"].isin(methods)]
    styles = _method_styles(methods)

    metrics = [
        ("power_FDR05", "tQTL power\n(BH FDR < 0.05)", None),
        ("type1_p05", "Type I error\n(nominal P < 0.05)", 0.05),
        ("empirical_fdr_FDR05", "Empirical FDR\n(BH FDR < 0.05)", 0.05),
    ]
    states = ["unconfounded", "confounded"]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True)
    letters = iter("abcdef")
    summary_parts = []

    for r, state in enumerate(states):
        for c, (metric, ylabel, ref) in enumerate(metrics):
            ax = axes[r, c]
            sub = z[z["confounding_status"].eq(state)]
            sm = _summary(sub, metric, ["base_method", "rho_true"])
            sm["metric"] = metric
            sm["confounding_status"] = state
            summary_parts.append(sm)
            _plot_series(ax, sm, methods, styles)
            if ref is not None:
                ax.axhline(ref, color="#777777", lw=0.8, ls=":", zorder=0)
            ax.set_xlabel(r"Residual autocorrelation, $\rho$")
            ax.set_ylabel(ylabel)
            ax.set_title("No confounding" if state == "unconfounded" else "Confounding present")
            ax.set_xticks(sorted(z["rho_true"].dropna().unique()))
            if metric == "power_FDR05":
                ax.set_ylim(-0.02, 1.02)
            else:
                ymax = max(0.12, float(sm["ci95_high"].max()) * 1.12 if len(sm) else 0.12)
                ax.set_ylim(-0.005, min(1.0, ymax))
            _clean_axes(ax)
            _panel_label(ax, next(letters))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=min(5, max(1, math.ceil(len(labels) / 2))),
            frameon=False,
            handlelength=2.2,
            columnspacing=1.1,
        )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.94, bottom=0.20, wspace=0.43, hspace=0.48)
    _save(fig, out_dir / "simulation_tqtl_primary")
    return pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()


def plot_covariate_ablation(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    base_methods = [m for m in DEFAULT_DYNAMIC_METHODS if m in set(df["base_method"])]
    z = df[
        df["base_method"].isin(base_methods)
        & df["covariate_mode"].isin(["adjusted", "nocov"])
        & df["confounding_status"].eq("confounded")
    ].copy()
    if z.empty:
        return pd.DataFrame()
    styles = _method_styles(base_methods)
    metrics = [
        ("power_FDR05", "tQTL power\n(BH FDR < 0.05)", None),
        ("type1_p05", "Type I error\n(nominal P < 0.05)", 0.05),
        ("empirical_fdr_FDR05", "Empirical FDR\n(BH FDR < 0.05)", 0.05),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharex=True)
    summary_parts = []
    for i, (metric, ylabel, ref) in enumerate(metrics):
        ax = axes[i]
        sm = _summary(z, metric, ["base_method", "covariate_mode", "rho_true"])
        sm["metric"] = metric
        summary_parts.append(sm)
        _plot_series(
            ax, sm, base_methods, styles,
            label_mode="method_cov",
            linestyle_by_covariate=True,
        )
        if ref is not None:
            ax.axhline(ref, color="#777777", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel(r"Residual autocorrelation, $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(z["rho_true"].dropna().unique()))
        if metric == "power_FDR05":
            ax.set_ylim(-0.02, 1.02)
        else:
            ymax = max(0.12, float(sm["ci95_high"].max()) * 1.12 if len(sm) else 0.12)
            ax.set_ylim(-0.005, min(1.0, ymax))
        _clean_axes(ax)
        _panel_label(ax, "abc"[i])
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate legend entries generated across panels/modes.
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.91, bottom=0.31, wspace=0.45)
    _save(fig, out_dir / "simulation_covariate_ablation")
    return pd.concat(summary_parts, ignore_index=True)


def plot_dynamic_secondary(
    df: pd.DataFrame,
    out_dir: Path,
    dynamic_methods: List[str] | None,
) -> pd.DataFrame:
    dynamic_methods = dynamic_methods or DEFAULT_DYNAMIC_METHODS
    z = df[
        (~df["method"].astype(str).str.endswith("__nocov"))
        & df["base_method"].isin(dynamic_methods)
    ].copy()
    metric0 = "dynamic_power_stagewise_FDR05"
    if metric0 not in z.columns or z[metric0].notna().sum() == 0:
        return pd.DataFrame()
    methods = _method_order(z, dynamic_methods)
    styles = _method_styles(methods)
    metrics = [
        ("dynamic_power_stagewise_FDR05", "Dynamic-QTL power", None),
        ("dynamic_type1_static_stagewise_FDR05", "Static tQTL → dynamic\nfalse-positive rate", 0.05),
        ("dynamic_type1_null_stagewise_FDR05", "Null → dynamic\nfalse-positive rate", 0.05),
        ("dynamic_empirical_fdr_stagewise_FDR05", "Dynamic empirical FDR", 0.05),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.1, 4.35), sharex=True)
    summary_parts = []
    for i, (metric, ylabel, ref) in enumerate(metrics):
        ax = axes.flat[i]
        sm = _summary(z, metric, ["base_method", "rho_true", "confounding_status"])
        sm["metric"] = metric
        summary_parts.append(sm)
        # To keep the secondary figure readable, encode confounding state by line style.
        for method in methods:
            color, marker = styles[method]
            gm = sm[sm["base_method"].eq(method)]
            for state, ls, fill in [
                ("unconfounded", "-", color),
                ("confounded", "--", "white"),
            ]:
                h = gm[gm["confounding_status"].eq(state)].sort_values("rho_true")
                if h.empty:
                    continue
                x = h["rho_true"].to_numpy(float)
                y = h["mean"].to_numpy(float)
                lo = h["ci95_low"].to_numpy(float)
                hi = h["ci95_high"].to_numpy(float)
                label = f"{METHOD_LABELS.get(method, method)} — {'confounded' if state == 'confounded' else 'no confounding'}"
                ax.plot(
                    x, y, color=color, marker=marker, linestyle=ls,
                    markerfacecolor=fill, markeredgecolor=color, markeredgewidth=0.8,
                    label=label, zorder=3,
                )
                ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
        if ref is not None:
            ax.axhline(ref, color="#777777", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel(r"Residual autocorrelation, $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(z["rho_true"].dropna().unique()))
        if metric == metric0:
            ax.set_ylim(-0.02, 1.02)
        else:
            ymax = max(0.12, float(sm["ci95_high"].max()) * 1.12 if len(sm) else 0.12)
            ax.set_ylim(-0.005, min(1.0, ymax))
        _clean_axes(ax)
        _panel_label(ax, "abcd"[i])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.25, wspace=0.46, hspace=0.50)
    _save(fig, out_dir / "simulation_dynamic_secondary")
    return pd.concat(summary_parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="simulation_summary_by_replicate.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--main-methods",
        default=None,
        help="Optional comma-separated method order/subset for the primary tQTL figure.",
    )
    p.add_argument(
        "--dynamic-methods",
        default=",".join(DEFAULT_DYNAMIC_METHODS),
        help="Comma-separated methods for the secondary dynamic-QTL figure.",
    )
    a = p.parse_args()

    _style_rcparams()
    inp = Path(a.input)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    df = _ensure_design_columns(df)
    main_methods = _parse_csv_list(a.main_methods)
    dynamic_methods = _parse_csv_list(a.dynamic_methods)

    tables = []
    t1 = plot_tqtl_primary(df, out_dir, main_methods)
    if not t1.empty:
        t1["figure"] = "tqtl_primary"
        tables.append(t1)
    t2 = plot_covariate_ablation(df, out_dir)
    if not t2.empty:
        t2["figure"] = "covariate_ablation"
        tables.append(t2)
    t3 = plot_dynamic_secondary(df, out_dir, dynamic_methods)
    if not t3.empty:
        t3["figure"] = "dynamic_secondary"
        tables.append(t3)

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            out_dir / "simulation_key_metrics_summary.csv", index=False
        )

    print(f"Figures written to: {out_dir}")


if __name__ == "__main__":
    main()
