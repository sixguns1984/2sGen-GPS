#!/usr/bin/env python3
"""Pre-specified upstream IV-gene union and downstream long-form joins.

All candidate membership comes from upstream files; downstream significance never expands it.
"""
import argparse,json
from pathlib import Path
import pandas as pd

def parse_spec(s):
    k,sep,v=s.partition("=")
    if not sep or not k or not v:raise ValueError("Expected name=/absolute/file.csv")
    return k,Path(v)

def read(path):
    if not path.exists():raise FileNotFoundError(path)
    x=pd.read_csv(path,low_memory=False)
    if not {"gene","lead_snp"}.issubset(x):raise ValueError(f"Missing gene/lead_snp in {path}")
    x=x.copy();x["gene"]=x.gene.astype(str);x["lead_snp"]=x.lead_snp.astype(str)
    return x

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--upstream",action="append",required=True,help="validated phenotype master, name=path")
    a.add_argument("--downstream",action="append",default=[],help="one downstream result master, name=path")
    a.add_argument("--out-dir",required=True)
    z=a.parse_args();out=Path(z.out_dir);out.mkdir(parents=True,exist_ok=True)
    upstream=[]
    for spec in z.upstream:
        name,path=parse_spec(spec);d=read(path)
        if "primary_downstream_candidate" in d:
            keep=d.primary_downstream_candidate.astype(str).str.lower().isin(["true","1"]);d=d[keep].copy()
        d=d.drop_duplicates(["gene","lead_snp"]);d.insert(0,"upstream_phenotype",name)
        upstream.append(d)
    long=pd.concat(upstream,ignore_index=True);long.to_csv(out/"validated_upstream_IV_gene_long.csv",index=False)
    group=long.groupby(["gene","lead_snp"],as_index=False).agg(n_upstream_phenotypes=("upstream_phenotype","nunique"),upstream_phenotypes=("upstream_phenotype",lambda s:";".join(sorted(set(s)))))
    group.to_csv(out/"validated_upstream_IV_gene_union.csv",index=False)
    alljoined=[];audit=[]
    for spec in z.downstream:
        name,path=parse_spec(spec);d=read(path)
        d.insert(0,"modality_trait",name)
        joined=d.merge(group,on=["gene","lead_snp"],how="inner",validate="many_to_one")
        joined.to_csv(out/f"matched_{name.replace('/','_')}.csv",index=False)
        alljoined.append(joined)
        audit.append({"modality_trait":name,"tested_rows":len(d),"matched_rows":len(joined),"unique_matched_pairs":len(joined[["gene","lead_snp"]].drop_duplicates())})
    if alljoined:pd.concat(alljoined,ignore_index=True,sort=False).to_csv(out/"cross_modal_IV_gene_long.csv",index=False)
    pd.DataFrame(audit,columns=["modality_trait","tested_rows","matched_rows","unique_matched_pairs"]).to_csv(out/"downstream_match_audit.csv",index=False)
    (out/"summary_manifest.json").write_text(json.dumps({"upstream":z.upstream,"downstream":z.downstream,"n_union_pairs":len(group),"n_union_genes":int(group.gene.nunique()),"note":"Exploratory association; downstream P values are not new causal proof. Apply modality-specific FDR to full pre-specified test universe."},indent=2))
    print(f"Upstream validated union: {len(group)} pairs, {group.gene.nunique()} genes; saved to {out}")
if __name__=="__main__":main()
