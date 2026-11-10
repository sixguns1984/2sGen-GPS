#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import bootstrap
bootstrap()

from gengps.io import read_table, write_table
from gengps.benchmark.registry import run_method


def load_toml(path):
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        import toml
        return toml.load(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    cfg = load_toml(a.config)
    q = cfg["qtl"]
    mc = dict(cfg.get("method", {}))
    mc.update({
        "covariates": q.get("covariates", []),
        "id_col": q.get("id_col", "participant_id"),
        "visit_col": q.get("visit_col", "visit"),
        "submitted_visit_col": q.get("submitted_visit_col", "followup_time"),
        "followup_col": q.get("followup_col", "followup_time"),
        "age_col": q.get("age_col", "age"),
        "required_visits": q.get("required_visits"),
    })
    if a.device:
        mc["device"] = a.device

    cov = read_table(q["cov"], index_col=0)
    expr = read_table(q["expr"], index_col=0)
    geno = read_table(q["geno"], index_col=0)
    cis = read_table(q["cis"])
    res = run_method(a.method, cov, expr, geno, cis, mc)
    write_table(res, a.out)


if __name__ == "__main__":
    main()
