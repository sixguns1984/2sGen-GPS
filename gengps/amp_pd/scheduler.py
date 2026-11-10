from __future__ import annotations
import os, traceback, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from .config import AmpPdPaths, implementation_tag
from .real_qtl import run_gene_chunk, make_gene_chunks, method_allowed_in_cohort, bfile_for_cohort
from .stage1_data import get_stage1_genotype_source, stage1_expression_genes
from .fdr import finalize_method_fdr

CPU_ONLY_REVISED_METHODS = set()  # lin2018_regression_cov uses the original GPU regression path


def parse_gpu_batch_map(spec, gpus, default_batch=2048):
    gpus = [int(x) for x in gpus]
    out = {g: int(default_batch) for g in gpus}
    if spec is None or str(spec).strip() == "":
        return out
    for item in str(spec).split(","):
        if not item.strip():
            continue
        gpu, b = item.split(":", 1)
        out[int(gpu)] = int(b)
    return out


def _parquet_first_value(path: Path, column: str):
    try:
        pf = pq.ParquetFile(path)
        if column not in pf.schema_arrow.names or pf.metadata.num_rows == 0:
            return None
        tab = pf.read_row_group(0, columns=[column])
        if len(tab) == 0:
            return None
        return tab.column(0)[0].as_py()
    except Exception:
        return None


def _chunk_reusable(path, method, data_mode="current_v89", stage1_cell_mode="primary"):
    """Whether a checkpoint chunk belongs to the current method+data implementation."""
    p = Path(path)
    if not p.exists():
        return False
    expected = implementation_tag(method, data_mode, stage1_cell_mode=stage1_cell_mode)
    if expected is None:
        return True
    return str(_parquet_first_value(p, "implementation_tag")) == str(expected)


def _final_result_complete(
    final_dir: Path, method: str, data_mode="current_v89", stage1_cell_mode="primary",
    fdr_strategy: str | None = None,
):
    """Strict but cheap check used to avoid rerunning already-finalized methods."""
    final_dir = Path(final_dir)
    all_results = final_dir / "all_results.parquet"
    summary = final_dir / "fdr_finalize_summary.json"
    if not all_results.exists() or not summary.exists():
        return False, None
    try:
        info = json.loads(summary.read_text())
        n_expected = int(info.get("n_pairs"))
        n_actual = int(pq.ParquetFile(all_results).metadata.num_rows)
        if n_actual != n_expected:
            return False, None
        if fdr_strategy is not None:
            got_strategy = str(info.get("fdr_strategy", "unknown"))
            if got_strategy != str(fdr_strategy):
                return False, None
        expected_tag = implementation_tag(method, data_mode, stage1_cell_mode=stage1_cell_mode)
        if expected_tag is not None:
            got = _parquet_first_value(all_results, "implementation_tag")
            if str(got) != str(expected_tag):
                return False, None
        return True, n_actual
    except Exception:
        return False, None


def _gpu_dynamic_worker(root, cohort, method, task_q, result_q, gpu_id,
                        kwargs, cis_path, initial_batch, min_batch, data_mode, stage1_cell_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
    except Exception:
        pass

    local_batch = int(initial_batch)
    while True:
        task = task_q.get()
        if task is None:
            break
        idx, genes, outfile = task
        if _chunk_reusable(outfile, method, data_mode, stage1_cell_mode=stage1_cell_mode):
            result_q.put({"gpu": gpu_id, "chunk": idx, "status": "exists",
                          "batch_size": local_batch, "outfile": outfile})
            continue

        batch = local_batch
        while True:
            try:
                p = AmpPdPaths(root=root)
                k = dict(kwargs)
                k["batch_size"] = int(batch)
                # Incompatible legacy chunks are overwritten in place only after
                # the new method has actually reached this chunk.
                run_gene_chunk(p, cohort, method, genes, outfile, device="cuda:0",
                               cis_path_override=cis_path, data_mode=data_mode,
                               stage1_cell_mode=stage1_cell_mode, **k)
                local_batch = batch
                result_q.put({"gpu": gpu_id, "chunk": idx, "status": "done",
                              "batch_size": batch, "outfile": outfile})
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg or ("cuda" in msg and "memory" in msg)) and batch > min_batch:
                    batch = max(int(min_batch), batch // 2)
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    result_q.put({"gpu": gpu_id, "chunk": idx, "status": "oom_retry",
                                  "batch_size": batch, "message": str(e)[:300]})
                    continue
                result_q.put({"gpu": gpu_id, "chunk": idx, "status": "error",
                              "batch_size": batch, "traceback": traceback.format_exc()})
                return
            except Exception:
                result_q.put({"gpu": gpu_id, "chunk": idx, "status": "error",
                              "batch_size": batch, "traceback": traceback.format_exc()})
                return

    result_q.put({"gpu": gpu_id, "status": "worker_finished", "batch_size": local_batch})


def _cpu_one(args):
    root, cohort, method, idx, genes, outfile, kwargs, cis_path, data_mode, stage1_cell_mode = args
    # One process per gene chunk; keep BLAS/OpenMP/Torch intra-process threading at
    # one to avoid severe oversubscription when cpu_workers > 1.
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    if _chunk_reusable(outfile, method, data_mode, stage1_cell_mode=stage1_cell_mode):
        return idx, "exists"
    p = AmpPdPaths(root=root)
    run_gene_chunk(p, cohort, method, genes, outfile, device="cpu",
                   cis_path_override=cis_path, data_mode=data_mode, stage1_cell_mode=stage1_cell_mode, **kwargs)
    return idx, "done"


def run_method_partitioned(
    paths: AmpPdPaths, cohort: str, method: str,
    cis_path=None, genes_per_chunk=40, gpus=(0, 1, 2), cpu_workers=12,
    batch_size=4096, gpu_batch_map=None, min_gpu_batch=512,
    rho_points=40, legacy_times=None, spline_df=5, variance_df=5,
    ctar_reference_interval=1.0, ctar_rho_max=0.99,
    genotype_mode="dosage", legacy_genotype_mode="dummies",
    dtype="float64", cell_adjust=False, result_label=None, finalize=True,
    temp_root="/public/labdata/luojunfeng/tmp", data_mode="current_v89",
    stage1_cell_mode="primary", fdr_strategy="stagewise_hierarchical", fdr_alpha=0.05,
):
    if not method_allowed_in_cohort(method, cohort):
        return {"skipped": True, "reason": "method/cohort not applicable"}
    if data_mode == "stage1_manuscript":
        if stage1_cell_mode not in {"primary", "nocell_audit"}:
            raise ValueError(f"Unknown stage1_cell_mode={stage1_cell_mode!r}")
        if stage1_cell_mode == "nocell_audit" and method != "submitted_followup_mptga":
            raise ValueError("Stage-I no-cell audit is defined only for submitted_followup_mptga")
        if cell_adjust:
            raise ValueError(
                "Stage-I now includes the six cell-composition covariates directly in the primary "
                "submitted MPTGA model; do not launch a separate --cell-adjust sensitivity analysis."
            )
        if str(legacy_genotype_mode) != "binary":
            raise ValueError(
                "Stage-I benchmark methods must use --legacy-genotype-mode binary to reproduce the "
                "original BinaryEncoder hard-call coding (0=00,1=01,2=10,-1=11)."
            )
        if method in {"submitted_followup_mptga", "lin2018_mptga"} and int(rho_points) != 100:
            raise ValueError("manuscript-exact MPTGA requires --rho-points 100 (rho=0.00,...,0.99)")

    if result_label is not None:
        label = result_label
    elif data_mode == "stage1_manuscript" and stage1_cell_mode == "nocell_audit":
        label = method + "__nocell_audit"
    else:
        label = method + ("__celladj" if cell_adjust else "")
    base = paths.qtl_dir / cohort / label
    chunk_dir = base / "chunks"
    final_dir = base / "final"
    log_dir = base / "logs"
    for d in [chunk_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Critical resumability rule: a successfully finalized method is never
    # recomputed merely because the user launches the all-method command again.
    if finalize:
        complete, n_done = _final_result_complete(
            final_dir, method, data_mode, stage1_cell_mode=stage1_cell_mode,
            fdr_strategy=fdr_strategy,
        )
        if complete:
            execution = {
                "cohort": cohort,
                "method": method,
                "result_label": label,
                "skipped": True,
                "reason": "already_finalized",
                "n_pairs": int(n_done),
                "final_dir": str(final_dir),
                "implementation_tag": implementation_tag(method, data_mode, stage1_cell_mode=stage1_cell_mode),
                "data_mode": data_mode,
                "stage1_cell_mode": stage1_cell_mode,
                "fdr_strategy": str(fdr_strategy),
                "fdr_alpha": float(fdr_alpha),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (log_dir / "execution_summary.json").write_text(json.dumps(execution, indent=2))
            return execution

    if cis_path is None:
        if data_mode == "stage1_manuscript":
            from .stage1_data import stage1_cache_paths
            cis_path = stage1_cache_paths(paths, "PP")["cis"]
        else:
            cis_path = paths.data_dir / "cis" / "cis_1000kb_PP.parquet"
    cis_path = str(cis_path)
    chunks = make_gene_chunks(paths, genes_per_chunk, cis_path, data_mode=data_mode)

    kwargs = dict(
        rho_points=rho_points, legacy_times=legacy_times,
        spline_df=spline_df, variance_df=variance_df,
        ctar_reference_interval=ctar_reference_interval,
        ctar_rho_max=ctar_rho_max,
        genotype_mode=genotype_mode,
        legacy_genotype_mode=legacy_genotype_mode,
        dtype=dtype, cell_adjust=cell_adjust,
    )
    tasks = [(i, g, str(chunk_dir / f"chunk_{i:05d}.parquet")) for i, g in enumerate(chunks)]
    pending = [t for t in tasks if not _chunk_reusable(t[2], method, data_mode, stage1_cell_mode=stage1_cell_mode)]
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cohort} {method}: "
        f"total_chunks={len(tasks)} pending={len(pending)} genes_per_chunk={int(genes_per_chunk)} "
        f"data_mode={data_mode}",
        flush=True,
    )

    execution = {
        "cohort": cohort,
        "method": method,
        "result_label": label,
        "dtype": dtype,
        "implementation_tag": implementation_tag(method, data_mode, stage1_cell_mode=stage1_cell_mode),
        "data_mode": data_mode,
        "stage1_cell_mode": stage1_cell_mode,
        "genes_per_chunk": int(genes_per_chunk),
        "requested_default_batch": int(batch_size),
        "gpus": [int(x) for x in gpus],
        "cell_adjust": bool(cell_adjust),
        "cis_path": cis_path,
        "temp_root": str(temp_root),
        "fdr_strategy": str(fdr_strategy),
        "fdr_alpha": float(fdr_alpha),
        "n_chunks_total": len(tasks),
        "n_chunks_pending_at_start": len(pending),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    force_cpu = method in CPU_ONLY_REVISED_METHODS
    if method == "manova" or force_cpu or not gpus:
        cpu_kwargs = dict(kwargs)
        cpu_kwargs["batch_size"] = int(batch_size)
        args = [(paths.root, cohort, method, i, g, o, cpu_kwargs, cis_path, data_mode, stage1_cell_mode) for i, g, o in pending]
        with ProcessPoolExecutor(max_workers=max(1, int(cpu_workers))) as ex:
            futs = [ex.submit(_cpu_one, a) for a in args]
            for f in as_completed(futs):
                f.result()
        execution["compute_mode"] = "cpu_forced_method" if force_cpu else "cpu"
        execution["cpu_workers"] = int(cpu_workers)
        execution["cpu_intraop_threads"] = 1
    else:
        gpus = [int(x) for x in gpus]
        batch_map = parse_gpu_batch_map(gpu_batch_map, gpus, batch_size)
        task_q = Queue()
        result_q = Queue()
        for t in pending:
            task_q.put(t)
        for _ in gpus:
            task_q.put(None)

        procs = []
        for gpu in gpus:
            pr = Process(
                target=_gpu_dynamic_worker,
                args=(paths.root, cohort, method, task_q, result_q, gpu, kwargs,
                      cis_path, batch_map[gpu], int(min_gpu_batch), data_mode, stage1_cell_mode),
            )
            pr.start()
            procs.append(pr)

        finished = 0
        events = []
        errors = []
        n_done_runtime = 0
        progress_t0 = time.time()
        while finished < len(procs):
            msg = result_q.get()
            events.append(msg)
            status = msg.get("status")
            if status == "worker_finished":
                finished += 1
            elif status in {"done", "exists"}:
                n_done_runtime += 1
                elapsed = max(time.time() - progress_t0, 1e-9)
                rate = n_done_runtime / elapsed
                remain = max(len(pending) - n_done_runtime, 0)
                eta_h = remain / max(rate, 1e-12) / 3600.0
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cohort} {method}: "
                    f"chunk={msg.get('chunk')} gpu={msg.get('gpu')} status={status} "
                    f"progress={n_done_runtime}/{len(pending)} "
                    f"rate={rate*3600:.2f} chunks/h eta={eta_h:.2f} h "
                    f"batch={msg.get('batch_size')}",
                    flush=True,
                )
            elif status == "oom_retry":
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cohort} {method}: "
                    f"GPU {msg.get('gpu')} OOM; retry batch={msg.get('batch_size')}",
                    flush=True,
                )
            elif status == "error":
                errors.append(msg)
                break

        if errors:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
            for pr in procs:
                pr.join()
            (log_dir / "gpu_events.json").write_text(json.dumps(events, indent=2))
            raise RuntimeError("GPU worker failure:\n" + "\n".join(e.get("traceback", "") for e in errors))

        for pr in procs:
            pr.join()
        (log_dir / "gpu_events.json").write_text(json.dumps(events, indent=2))
        execution["compute_mode"] = "dynamic_gpu_queue"
        execution["gpu_batch_map"] = batch_map
        execution["n_oom_retries"] = sum(1 for e in events if e.get("status") == "oom_retry")
        execution["chunks_by_gpu"] = {
            str(g): sum(1 for e in events if e.get("gpu") == g and e.get("status") == "done")
            for g in gpus
        }

    # Stale-chunk guard: finalization must see exactly the current task set.
    expected_paths = {str(Path(o).resolve()) for _, _, o in tasks}
    stale = []
    for cp in chunk_dir.glob("chunk_*.parquet"):
        if str(cp.resolve()) not in expected_paths:
            stale.append(cp)
    if stale:
        qdir = base / ("stale_chunks_" + time.strftime("%Y%m%d_%H%M%S"))
        qdir.mkdir(parents=True, exist_ok=True)
        for cp in stale:
            cp.rename(qdir / cp.name)
        execution["stale_chunks_quarantined"] = [str(x) for x in stale]
        execution["stale_chunk_quarantine_dir"] = str(qdir)

    bad_expected = [o for _, _, o in tasks if not _chunk_reusable(o, method, data_mode, stage1_cell_mode=stage1_cell_mode)]
    if bad_expected:
        raise RuntimeError(
            f"{len(bad_expected)} expected QTL chunks are missing/incompatible after compute; "
            f"first={bad_expected[:5]}"
        )

    n = None
    if finalize:
        n = finalize_method_fdr(
            str(chunk_dir), str(final_dir), temp_root=temp_root,
            work_label=f"{cohort}__{label}",
            # Never reuse a legacy DuckDB base when switching Stage-I data source.
            reuse_legacy_work_db=(data_mode != "stage1_manuscript"),
            fdr_strategy=fdr_strategy,
            alpha=fdr_alpha,
        )
    execution["n_pairs"] = n
    execution["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (log_dir / "execution_summary.json").write_text(json.dumps(execution, indent=2))
    return execution


def _bim_snps_for_cohort(paths: AmpPdPaths, cohort: str):
    bfile = bfile_for_cohort(paths, cohort)
    bim = pd.read_csv(str(bfile) + ".bim", sep=r"\s+", header=None, usecols=[1], dtype=str)
    return set(bim.iloc[:, 0].astype(str))


def build_replication_cis_from_discovery(
    paths: AmpPdPaths, method: str, replication_cohort: str, data_mode="current_v89"
):
    # Replication tests the PPMI discovery-significant temporal tQTL pairs only.
    final_dir = paths.qtl_dir / "PP" / method / "final"
    sig = final_dir / "significant_tqtl.parquet"
    if not sig.exists():
        sig = final_dir / "significant_temporal.parquet"
    if not sig.exists():
        raise FileNotFoundError(sig)
    z = pd.read_parquet(sig, columns=["Variant_ID", "Ensembl_ID"]).drop_duplicates()

    if data_mode == "stage1_manuscript":
        if str(replication_cohort) != "PD":
            raise ValueError("Manuscript Stage-I replication cohort is PDBP (PD) only")
        gsrc = get_stage1_genotype_source(paths, "PD")
        uniq = z["Variant_ID"].astype(str).drop_duplicates().tolist()
        available = set()
        step = 50000
        for st in range(0, len(uniq), step):
            available.update(gsrc.available_snps(uniq[st:st + step]))
        genes = stage1_expression_genes(paths, "PD")
        z = z[
            z["Variant_ID"].astype(str).isin(available)
            & z["Ensembl_ID"].astype(str).isin(genes)
        ].copy()
    else:
        target_snps = _bim_snps_for_cohort(paths, replication_cohort)
        z = z[z["Variant_ID"].astype(str).isin(target_snps)].copy()

    outdir = paths.qtl_dir / "replication_maps"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "stage1_manuscript" if data_mode == "stage1_manuscript" else "current_v89"
    out = outdir / f"{method}_PP_hits_for_{replication_cohort}__{suffix}.parquet"
    z.rename(columns={"Variant_ID": "snpid", "Ensembl_ID": "geneid"}).to_parquet(
        out, index=False, compression="zstd"
    )
    return out, len(z)
