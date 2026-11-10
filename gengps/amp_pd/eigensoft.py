from __future__ import annotations
from pathlib import Path
import json, re, subprocess, os, shutil
import numpy as np
import pandas as pd

DEFAULT_TWSTATS = "twstats"
DEFAULT_TWTABLE = "twtable"


def parse_map(spec: str, default=None):
    out = dict(default or {})
    if spec is None or str(spec).strip() == "":
        return out
    for item in str(spec).split(","):
        if not item.strip():
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = int(v)
    return out


def _read_twstats_table(path: Path) -> pd.DataFrame:
    """Parse EIGENSOFT twstats output on the user's server.

    Expected layout:
      #N eigenvalue difference twstat p-value effect. n
       1 12.616700 NA 3.980 0.000230076 33.720
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = re.split(r"\s+", s)
        if len(toks) < 5:
            continue
        try:
            pc = int(float(toks[0]))
        except Exception:
            continue

        # Known EIGENSOFT output: the 5th token is p-value.
        try:
            p = float(toks[4])
            if np.isfinite(p) and 0.0 <= p <= 1.0:
                rows.append((pc, p))
                continue
        except Exception:
            pass

        # Conservative fallback: choose the rightmost probability-like value.
        nums = []
        for tok in toks:
            try:
                nums.append(float(tok))
            except Exception:
                nums.append(np.nan)
        p = None
        for x in nums[::-1]:
            if np.isfinite(x) and 0.0 <= x <= 1.0:
                p = float(x)
                break
        if p is not None:
            rows.append((pc, p))

    if not rows:
        raise ValueError(f"Could not parse Tracy-Widom p-values from {path}")
    return (pd.DataFrame(rows, columns=["PC", "p_value"])
            .drop_duplicates("PC", keep="first")
            .sort_values("PC")
            .reset_index(drop=True))


def summarize_significant_pcs(tw: pd.DataFrame, alpha=0.05, max_pcs=20):
    """Select the leading consecutive statistically significant PCs.

    This matches the submitted analysis logic: EIGENSTRAT/twstats determines
    how many *initial* PCs are significant, and PLINK PC1..PCK are then used as
    covariates.  Later isolated nominally significant PCs after the first
    non-significant PC are retained only as diagnostics and do not cause the
    search range to expand.

    Example:
      PC1,2,3 significant; PC4 non-significant; PC8,9 significant
      -> K=3 and selected PCs = [1,2,3]
      -> isolated later PCs [8,9] are reported diagnostically.
    """
    z = tw.sort_values("PC").copy()
    z = z[z["PC"].astype(int) <= int(max_pcs)].copy()
    finite = z[np.isfinite(z["p_value"])].copy()

    all_sig = finite.loc[
        finite["p_value"].astype(float) < float(alpha), "PC"
    ].astype(int).tolist()

    leading = []
    pmap = dict(zip(finite["PC"].astype(int), finite["p_value"].astype(float)))
    for pc in range(1, int(max_pcs) + 1):
        p = pmap.get(pc, np.nan)
        if np.isfinite(p) and p < float(alpha):
            leading.append(pc)
        else:
            break

    later_sig = [pc for pc in all_sig if pc not in set(leading)]
    return {
        "selected_n_pcs": int(len(leading)),
        "selected_pc_indices": leading,
        "selection_rule": "leading_consecutive_twstats_p_lt_alpha",
        "n_significant_leading": int(len(leading)),
        "all_nominal_significant_pc_indices": all_sig,
        "later_isolated_significant_pc_indices": later_sig,
        "nonconsecutive_significance_observed": bool(later_sig),
        "n_finite_twstats_pvalues": int(len(finite)),
    }

def resolve_twstats_command(twstats: str):
    """Validate twstats while preserving the activated environment's PATH."""
    twstats = str(twstats)
    if "/" in twstats:
        p = Path(twstats).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"twstats executable not found: {p}")
        if not os.access(str(p), os.X_OK):
            raise PermissionError(f"twstats is not executable: {p}")
        return twstats, str(p.resolve())

    resolved = shutil.which(twstats)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find '{twstats}' on PATH. Run `which twstats` in the same activated environment."
        )
    return twstats, resolved


def run_twstats(eigenval_file: str, out_file: str,
                twstats: str = DEFAULT_TWSTATS,
                twtable: str = DEFAULT_TWTABLE):
    """Run the exact command proven to work interactively.

      twstats -t twtable -i <PLINK .pca.eigenval> -o <output>

    No cwd change and no hard-coded old EIGENSOFT binary. The subprocess
    inherits PATH/LD_LIBRARY_PATH from the activated scvi environment.
    """
    eigenval_path = Path(eigenval_file).expanduser().resolve()
    out_path = Path(out_file).expanduser().resolve()
    if not eigenval_path.is_file():
        raise FileNotFoundError(f"PLINK eigenvalue file not found: {eigenval_path}")

    exe_arg, resolved = resolve_twstats_command(twstats)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe_arg, "-t", str(twtable), "-i", str(eigenval_path), "-o", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=os.environ.copy())
    log_path = Path(str(out_path) + ".log")
    log_path.write_text(proc.stdout or "", encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            "EIGENSOFT twstats failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"resolved twstats on PATH: {resolved}\n"
            f"return code: {proc.returncode}\n"
            f"log: {log_path}\n"
            f"output:\n{proc.stdout}"
        )
    if not out_path.exists():
        raise RuntimeError(
            "twstats returned exit code 0 but did not create the requested output.\n"
            f"command: {' '.join(cmd)}\nresolved twstats on PATH: {resolved}\nlog: {log_path}"
        )

    return {
        "command": cmd,
        "resolved_twstats_on_PATH": resolved,
        "twstats_argument": exe_arg,
        "twtable_argument": str(twtable),
        "eigenval_file": str(eigenval_path),
        "twstats_out": str(out_path),
        "twstats_log": str(log_path),
    }


def select_genotype_pc_count(qc_bfile: str, group_label: str,
                             pc_mode="eigensoft", manual_counts=None,
                             twstats: str = DEFAULT_TWSTATS,
                             twtable: str = DEFAULT_TWTABLE,
                             alpha=0.05, max_pcs=20):
    """Select genotype PCs using PLINK eigenvalues + PATH twstats."""
    manual_counts = manual_counts or {}
    qc_bfile = str(qc_bfile)
    outdir = Path(qc_bfile).parent / "eigensoft"
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "group": group_label,
        "mode_requested": pc_mode,
        "alpha": float(alpha),
        "max_pcs_tested": int(max_pcs),
        "selection_method": "PLINK PCA20 + PATH twstats; retain leading consecutive significant PCs",
    }

    if pc_mode == "manual":
        n = int(manual_counts[group_label])
        indices = list(range(1, n + 1))
        meta.update({"selected_n_pcs": n, "selected_pc_indices": indices,
                     "selection_source": "manual_debug_only"})
        return n, meta

    try:
        eigenval = Path(qc_bfile + ".pca.eigenval")
        twout = outdir / f"{group_label}.twstats.out"
        run_meta = run_twstats(str(eigenval), str(twout), twstats=twstats, twtable=twtable)

        tw = _read_twstats_table(twout)
        tw = tw[tw["PC"].astype(int) <= int(max_pcs)].copy()
        tw["significant"] = tw["p_value"].astype(float) < float(alpha)
        parsed = outdir / f"{group_label}.twstats.parsed.csv"
        tw.to_csv(parsed, index=False)

        stat = summarize_significant_pcs(tw, alpha=alpha, max_pcs=max_pcs)
        n = int(stat["selected_n_pcs"])
        if n < 1:
            raise RuntimeError(f"twstats selected 0 genotype PCs for {group_label}; inspect {parsed}.")

        meta.update(run_meta)
        meta.update(stat)
        meta.update({
            "selected_n_pcs": n,
            "selection_source": "PATH_twstats_on_PLINK_eigenvalues",
            "twstats_parsed": str(parsed),
            "p_values": tw[["PC", "p_value", "significant"]].to_dict(orient="records"),
        })
        if stat["nonconsecutive_significance_observed"]:
            meta["warning"] = (
                "Later isolated Tracy-Widom-significant PCs were observed after the first "
                "non-significant leading PC. They are reported diagnostically but are not "
                "included in the model. The production rule retains only the leading "
                "consecutive significant PCs, PC1..PCK."
            )
        return n, meta

    except Exception as e:
        meta["eigensoft_error"] = repr(e)
        if pc_mode == "eigensoft":
            raise RuntimeError(
                f"Could not determine genotype PCs for {group_label} using PLINK eigenvalues + PATH twstats.\n"
                f"{json.dumps(meta, indent=2)}"
            ) from e

    n = int(manual_counts[group_label])
    indices = list(range(1, n + 1))
    meta.update({"selected_n_pcs": n, "selected_pc_indices": indices,
                 "selection_source": "manual_fallback_debug_only"})
    return n, meta
