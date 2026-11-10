#!/usr/bin/env python3
"""Figure 2 from POWER80-filtered three-phenotype PD table.

Panel a: A-tier associations for UPDRS I/II/III only.
Panels b-c: two biologically interpretable example trajectories from frozen Stage II caches.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from helpers_pd_figure2 import (
    setup_rcparams, read_publication_table, attach_artifact_ids,
    is_a_tier, detect_story_examples, panel_a_order, draw_panel_a)
from component_trajectories import load_example_from_stage2, plot_example
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.panel_a_label_utils import select_panel_a_labels, write_label_audit
from common.editable_export import save_figure

DEFAULT_ROOT='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'


def _find_pub(root: Path):
    candidates = [
        root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'Supplement_PD_IV_gene_publication_3phenotypes_POWER80.csv',
        root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'Supplement_PD_IV_gene_publication_3phenotypes_POWER80_v7.csv',
    ]
    candidates += list((root/'single_iv_causal').glob('**/Supplement_PD_IV_gene_publication_3phenotypes_POWER80*.csv'))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError('Missing Supplement_PD_IV_gene_publication_3phenotypes_POWER80*.csv under single_iv_causal')


def _find_artifact(root: Path):
    candidates = [
        root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'PD_validated_IV_gene_stage1_candidates.csv',
        root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'PD_validated_IV_gene_stage1_candidates_POWER80.csv',
    ]
    candidates += list((root/'single_iv_causal').glob('**/PD_validated_IV_gene_stage1_candidates*.csv'))
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_stage1_bim_alleles(bim_path, variants):
    """Read exact rsIDs from the PPMI Stage-I BIM, without guessing dosage orientation.

    An allele pair must have one unique BIM row; the plotted Stage-II dosage does
    NOT automatically denote copies of A1 or copies of A2.
    """
    bim_path = Path(bim_path)
    if not bim_path.is_file():
        raise FileNotFoundError(
            f"PPMI Stage-I allele BIM missing: {bim_path}. "
            "Specify the actual file with --bim. No allele annotation was invented."
        )
    wanted = {str(x).strip() for x in variants}
    records = {}
    with bim_path.open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6 or fields[1] not in wanted:
                continue
            snp = fields[1]
            if snp in records:
                raise ValueError(f"Non-unique SNP {snp} in Stage-I BIM: {bim_path}")
            a1, a2 = fields[4].upper(), fields[5].upper()
            if not a1 or not a2 or a1 in ('.', '0') or a2 in ('.', '0'):
                raise ValueError(f"Missing BIM alleles for {snp}: {a1}/{a2}")
            records[snp] = (a1, a2)
    absent = sorted(wanted - records.keys())
    if absent:
        raise ValueError(f"Missing exact SNP IDs in Stage-I BIM {bim_path}: {absent}")
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=DEFAULT_ROOT)
    p.add_argument('--publication-table',default='')
    p.add_argument('--artifact-table',default='')
    p.add_argument('--outdir',default='')
    p.add_argument('--story-examples',nargs='*',default=[],help='Optional gene:phenotype selections; otherwise rank from NEW POWER80 results')
    p.add_argument('--example-cohort',choices=['PP','PD','SU','ALL'],default='PP')
    p.add_argument('--year-bin',type=float,default=1.0)
    p.add_argument('--top-labels-per-phenotype',type=int,default=8)
    p.add_argument('--panel-a-only',action='store_true')
    p.add_argument('--story-csv',default='')
    p.add_argument('--freq-file',default='',help='Legacy argument; plotted-sample MAF is computed from Stage-II hardcall donors instead')
    p.add_argument('--bim',default='',help='PPMI Stage-I BIM for exact PLINK A1/A2 (default ROOT/data/genotype/PP/amp_pd_PP_qc.bim)')
    p.add_argument('--example-title-note',default='Power-filtered (PPMI and PDBP bootstrap power ≥80%)')
    for ph in ('updrs1','updrs2','updrs3'):
        p.add_argument(f'--phenotype-{ph}',default='')
    a=p.parse_args(); setup_rcparams()
    root=Path(a.root)
    pub=Path(a.publication_table) if a.publication_table else _find_pub(root)
    artifact = Path(a.artifact_table) if a.artifact_table else _find_artifact(root)
    out=Path(a.outdir) if a.outdir else root/'manuscript_figures'/'tqtl_benchmark_v4'/'figure2_pd_power80'
    out.mkdir(parents=True,exist_ok=True)

    df_all=read_publication_table(pub)
    if artifact is not None:
        df_all=attach_artifact_ids(df_all, str(artifact))
    df=df_all[df_all['descriptive_evidence_tier'].map(is_a_tier)].copy()
    if df.empty:
        raise RuntimeError('No A-tier records in power-filtered PD table')
    if not df_all[['phenotype','gene','lead_snp']].drop_duplicates().shape[0] == len(df_all):
        raise ValueError('Duplicate phenotype + gene + SNP in POWER80 table')

    pdata=panel_a_order(df, a.top_labels_per_phenotype)
    ymax=max(3.2,float(pdata['neglog10p'].quantile(.97))*1.32)
    pdata=select_panel_a_labels(
        pdata, gene_col='gene', x_col='x', y_col='neglog10p',
        phenotype_col='phenotype', source_tier_col='descriptive_evidence_tier',
        cross_phenotype_col='shared_gene', y_cap=ymax*.89,
        top_n_per_group=a.top_labels_per_phenotype,
        top_group_order=[ph for ph in ('updrs1','updrs2','updrs3')
                         if ph in pdata['phenotype'].values],
    )
    pdata['label']=pdata['gene'].where(pdata['panel_a_label'], '')
    pdata['label_reason']=pdata['panel_a_label_reason']
    write_label_audit(pdata, out/'Figure2A_power80_label_audit.csv',
                      phenotype_col='phenotype', extra_cols=(
                        'lead_snp', 'selected_replication_pvalue',
                        'multiIV_p_q_BH', 'label', 'label_reason'))

    if a.story_csv:
        selected=pd.read_csv(a.story_csv)
        pieces=[]
        for _,q in selected.iterrows():
            z=df_all[(df_all.gene.astype(str)==str(q.gene)) & (df_all.lead_snp.astype(str)==str(q.lead_snp)) & (df_all.phenotype.astype(str)==str(q.phenotype).lower())]
            if len(z)!=1: raise ValueError(f'Cannot uniquely match story row {q.to_dict()}')
            parts=z.iloc[0].to_dict()
            # Do NOT trust story-CSV MAF/alleles: annotate from plotted donors and Stage-I BIM.
            pieces.append(parts)
        story=pd.DataFrame(pieces)
    else:
        story=detect_story_examples(df, requested=a.story_examples, n_examples=2)
    if len(story) != 2:
        raise RuntimeError(f'Figure 2 requires exactly 2 verified examples, got {len(story)}')
    # Annotation is completed from actual plotting participants below; do not save stale frequency values.
    bim_path=Path(a.bim) if a.bim else root/'data'/'genotype'/'PP'/'amp_pd_PP_qc.bim'
    allele_lookup=load_stage1_bim_alleles(bim_path, story['lead_snp'].tolist()) if not a.panel_a_only else {}

    # panel A standalone
    figA,axA=plt.subplots(figsize=(10.0,5.0))
    axA.set_ylim(0,ymax); draw_panel_a(axA,pdata)
    axA.set_title('a  A-tier temporal IV–gene–phenotype associations',loc='left',weight='bold',pad=10)
    figA.tight_layout(rect=[0,0,1,.96])
    for ext in ['pdf','svg','png']:
        save_figure(figA, out/f'Figure2A_PD_A_tier_power80.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(figA)
    if a.panel_a_only:
        print('[DONE] Panel A only:', out)
        return

    # Non-uniform Nature-style montage: compact panel a with two smaller story examples on the right,
    # leaving extra whitespace for the user to append additional subpanels later.
    fig=plt.figure(figsize=(11.8,7.7))
    gs=fig.add_gridspec(100,100,left=0.035,right=0.985,top=0.955,bottom=0.08)

    # Panel a deliberately compact; right-side examples are narrower and slightly taller.
    ax=fig.add_subplot(gs[10:64, 3:49])
    ax.set_ylim(0,ymax)
    draw_panel_a(ax,pdata,show_legend=True)
    ax.set_title('a  A-tier temporal IV–gene–phenotype associations',loc='left',weight='bold',pad=9)
    ax.text(0.0,1.055,a.example_title_note,transform=ax.transAxes,ha='left',va='bottom',fontsize=7.2,color='#607080')

    audit=[]
    specs=[gs[8:46, 64:94], gs[55:93, 64:94]]
    letters=['b','c']
    cohort_label={'PP':'PPMI','PD':'PDBP','SU':'SURE-PD3','ALL':'pooled'}[a.example_cohort]
    from component_trajectories import plotted_donor_genotype_summary
    story=story.reset_index(drop=True)
    for i,(_,row) in enumerate(story.iterrows()):
        z,metadata=load_example_from_stage2(root,row,cohort=a.example_cohort,
            phenotype_override=getattr(a,'phenotype_'+row['phenotype']))
        snp=str(row['lead_snp'])
        a1, a2=allele_lookup[snp]
        genotype_info=plotted_donor_genotype_summary(z)
        plot_example(fig,specs[i],row,z,letters[i],year_bin=a.year_bin,
                     cohort_label=cohort_label, bim_alleles=(a1,a2))
        # Explicit per-story provenance; the MAF always reflects plotted donors, not all WGS.
        metadata.update({
            'panel':letters[i], 'maf_plotted':genotype_info['maf_plotted'],
            'maf_source':'unique plotted donor Stage-II hardcall iv_dosage',
            'n_dosage0':genotype_info['n0'], 'n_dosage1':genotype_info['n1'],
            'n_dosage2':genotype_info['n2'],
            'bim_source':str(bim_path), 'bim_a1':a1, 'bim_a2':a2,
            'allele_dosage_orientation':'Stage-II dosage orientation not assumed',
        })
        story.loc[i,'maf_plotted']=genotype_info['maf_plotted']
        story.loc[i,'maf_PP']=genotype_info['maf_plotted'] if a.example_cohort=='PP' else float('nan')
        story.loc[i,'alleles_PP']=f'{a1}/{a2}'
        story.loc[i,'BIM_A1']=a1
        story.loc[i,'BIM_A2']=a2
        for dosage in range(3):
            story.loc[i,f'plotted_dosage{dosage}_n']=genotype_info[f'n{dosage}']
        audit.append(metadata)
        print(f'[EXAMPLE] {metadata["gene"]}/{metadata["lead_snp"]} {metadata["phenotype"]}: '
              f'{metadata["n_participants"]} participants; genotype counts={metadata["genotype_counts"]}; '
              f'plotted MAF={genotype_info["maf_plotted"]:.3f}; PLINK A1/A2={a1}/{a2}; '
              f'component={metadata["component"]}', flush=True)
    story[[c for c in ['phenotype','gene','lead_snp','artifact_id','selected_lag_order',
                       'descriptive_evidence_tier','PD_PLUS_SU_pvalue','PPMI_coef_sum',
                       'PD_PLUS_SU_coef_sum','maf_plotted','maf_PP','alleles_PP','BIM_A1',
                       'BIM_A2','plotted_dosage0_n','plotted_dosage1_n','plotted_dosage2_n']
           if c in story.columns]].to_csv(out/'selected_story_examples_power80.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'Figure2_examples_allele_MAF_audit.csv',index=False)

    fig.text(.50,.024,
             'The summed-lag coefficient describes an aggregate visit-based association, not a fixed-year causal effect. '
             'Only UPDRS I/II/III candidates passing PPMI/PDBP bootstrap screening are shown.',
             ha='center',va='bottom',fontsize=7.6,color='#586473')
    for ext in ['pdf','svg','png']:
        save_figure(fig, out/f'Figure2_PD_temporal_association_main_power80.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(audit).to_json(out/'Figure2_examples_data_provenance_power80.json',orient='records',indent=2)
    print('[DONE] Full PD Figure 2 power80:', out)

if __name__=='__main__':
    main()
