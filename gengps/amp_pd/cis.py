from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import AmpPdPaths, GENOTYPE_QC_GROUPS


def load_bim(bfile_prefix: str):
    b = pd.read_csv(
        str(bfile_prefix) + ".bim",
        sep=r"\s+",
        header=None,
        names=["chr", "snpid", "cm", "pos", "a1", "a2"],
        dtype={"chr": str, "snpid": str},
    )
    b["chr"] = b["chr"].astype(str).str.replace(r"^chr", "", regex=True)
    b["pos"] = pd.to_numeric(b["pos"], errors="coerce")
    return b.dropna(subset=["pos", "snpid"]).copy()


def bfile_for_group(paths: AmpPdPaths, group_label: str):
    return paths.data_dir / "genotype" / group_label / f"amp_pd_{group_label}_qc"


def build_cis_map(gene_locations: pd.DataFrame, bim: pd.DataFrame, window=1_000_000):
    rows = []
    for chrom, gc in gene_locations.groupby(gene_locations["chr"].astype(str)):
        vc = bim[bim["chr"].astype(str) == str(chrom)].sort_values("pos").reset_index(drop=True)
        if vc.empty:
            continue
        pos = vc["pos"].to_numpy(np.int64)
        ids = vc["snpid"].astype(str).to_numpy()
        for _, g in gc.iterrows():
            lo = max(0, int(g["start"]) - int(window))
            hi = int(g["end"]) + int(window)
            l = np.searchsorted(pos, lo, side="left")
            r = np.searchsorted(pos, hi, side="right")
            if r <= l:
                continue
            rows.extend(
                (str(g["gene_name"]), str(g.get("gene_id", "")), str(chrom),
                 int(g["start"]), int(g["end"]), str(ids[j]), int(pos[j]))
                for j in range(l, r)
            )
    return pd.DataFrame(
        rows,
        columns=["geneid", "ensembl_gene_id", "chr", "gene_start", "gene_end", "snpid", "snp_pos"],
    )


def prepare_cis_map(paths: AmpPdPaths, window=1_000_000):
    """Build separate discovery and replication cis maps.

    Discovery uses the PPMI-specific post-QC BIM. Replication uses the
    PDBP+BioFIND post-QC BIM. The default qtl code uses the PPMI map for
    discovery and then filters PPMI hits to the target replication BIM.
    """
    ann = pd.read_parquet(paths.data_dir / "annotation" / "gene_locations.parquet")
    outdir = paths.data_dir / "cis"
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    cis_pp = None

    for group in ["PP", "REP"]:
        bfile = bfile_for_group(paths, group)
        bim = load_bim(str(bfile))
        cis = build_cis_map(ann, bim, window)
        out = outdir / f"cis_{window // 1000}kb_{group}.parquet"
        cis.to_parquet(out, index=False, compression="zstd")
        per_gene = cis.groupby("geneid").size().rename("n_cis_snps").reset_index()
        per_gene.to_csv(outdir / f"cis_{window // 1000}kb_{group}_per_gene.csv", index=False)
        summaries.append({
            "group": group,
            "bfile": str(bfile),
            "cis_path": str(out),
            "cis_pairs": int(len(cis)),
            "genes": int(cis["geneid"].nunique()) if len(cis) else 0,
            "snps": int(cis["snpid"].nunique()) if len(cis) else 0,
        })
        if group == "PP":
            cis_pp = cis

    # Backward-compatible discovery alias.
    if cis_pp is not None:
        cis_pp.to_parquet(outdir / f"cis_{window // 1000}kb.parquet", index=False, compression="zstd")

    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / f"cis_{window // 1000}kb_summary.csv", index=False)
    return summary
