from __future__ import annotations

import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .registry import ALL_METHODS, GPU_METHODS


def _base_method(name: str) -> str:
    return name[:-len("__nocov")] if name.endswith("__nocov") else name


def schedule_commands(config_path, methods, out_dir, gpus=None, cpu_parallel=2, resume=True):
    """Run benchmark methods with fail-fast subprocess handling.

    `gpus=[]` is the supported CPU-only mode.  Any failed method raises with the
    captured stdout/stderr instead of silently returning an empty benchmark row.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = list(methods or ALL_METHODS)
    gpus = [] if gpus is None else [str(x) for x in gpus if str(x) != ""]
    script = Path(__file__).resolve().parents[2] / "scripts_benchmark" / "run_one_qtl_method.py"

    jobs = []
    for m in methods:
        outfile = out / f"{m}.parquet"
        if resume and outfile.exists():
            jobs.append((m, outfile, "existing"))
        else:
            jobs.append((m, outfile, "run"))

    pending = [(m, o) for m, o, status in jobs if status == "run"]
    cpu = [x for x in pending if _base_method(x[0]) not in GPU_METHODS or not gpus]
    gpu = [x for x in pending if x not in cpu]

    def run(cmd, env=None):
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "Benchmark subprocess failed\n"
                f"command: {' '.join(map(str, cmd))}\n"
                f"stdout:\n{p.stdout}\n"
                f"stderr:\n{p.stderr}\n"
            )
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

    results = {m: {"status": "existing", "out": str(o)} for m, o, status in jobs if status == "existing"}
    with ThreadPoolExecutor(max_workers=max(1, int(cpu_parallel) + len(gpus))) as ex:
        futs = {}
        for m, o in cpu:
            cmd = [
                sys.executable, str(script), "--config", str(config_path),
                "--method", m, "--out", str(o), "--device", "cpu",
            ]
            futs[ex.submit(run, cmd, os.environ.copy())] = (m, o)

        if gpu:
            q = queue.Queue()
            for x in gpu:
                q.put(x)

            def gpu_worker(gpu_id):
                rr = []
                while True:
                    try:
                        m, o = q.get_nowait()
                    except queue.Empty:
                        break
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    env.setdefault("OMP_NUM_THREADS", "1")
                    env.setdefault("MKL_NUM_THREADS", "1")
                    env.setdefault("OPENBLAS_NUM_THREADS", "1")
                    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
                    cmd = [
                        sys.executable, str(script), "--config", str(config_path),
                        "--method", m, "--out", str(o), "--device", "cuda:0",
                    ]
                    rr.append((m, o, run(cmd, env)))
                    q.task_done()
                return rr

            for gi in gpus:
                futs[ex.submit(gpu_worker, gi)] = (f"gpu{gi}", None)

        for f in as_completed(futs):
            tag, outp = futs[f]
            value = f.result()  # propagate failures immediately
            if tag.startswith("gpu"):
                for m, o, rr in value:
                    results[m] = {"status": "completed", "out": str(o), **rr}
            else:
                results[tag] = {"status": "completed", "out": str(outp), **value}

    # Output existence is part of successful completion.
    missing = [m for m in methods if not (out / f"{m}.parquet").exists()]
    if missing:
        raise RuntimeError(f"Benchmark methods completed without output parquet: {missing}")
    return results
