from __future__ import annotations
import json,time
import numpy as np,pandas as pd,torch
from scipy.stats import f as fdist
from ..basis import TimeBasis
from ..covariance import fit_smooth_ct_covariance,whiten_grouped,prepare_grouped_time_cache
from ..utils import bh_fdr,genotype_channels,participant_groups,require_aligned,stagewise_dynamic_fdr
from ..gpu import get_device,fwl_batch,configure_cuda_numerics

class AgeTQTLScanner:
    """Proposed age-based tQTL model.

    Mean:
      E = common age curve + covariates + genotype x age-curve + error

    Covariance:
      Cov(e_ij,e_ik) =
        sigma(age_ij) sigma(age_ik) exp(-phi |age_ij-age_ik|)

    log sigma^2(age) is a smooth cubic B-spline.
    No equal number of visits and no hard age bins are required.
    """
    def __init__(self,basis_kind="cubic",spline_df=5,variance_df=5,
                 genotype_mode="dosage",batch_size=4096,device=None,dtype="float64"):
        self.basis_kind=basis_kind;self.spline_df=spline_df;self.variance_df=variance_df
        self.genotype_mode=genotype_mode;self.batch_size=int(batch_size);self.device=get_device(device);configure_cuda_numerics(self.device,allow_tf32=False)
        self.dtype=torch.float64 if dtype=="float64" else torch.float32

    def scan(self,cov,expression,genotype,cis_map,covariate_cols=None,
             id_col="participant_id",age_col="age"):
        t0=time.time();require_aligned(cov,expression);covariate_cols=covariate_cols or []
        age=cov[age_col].to_numpy(float);groups=participant_groups(cov,id_col)
        grouped_cache=prepare_grouped_time_cache(age,groups)
        basis=TimeBasis(self.basis_kind,df=self.spline_df).fit(age);B=basis.transform(age)
        C=cov[covariate_cols].to_numpy(float) if covariate_cols else np.empty((len(cov),0))
        X0=np.column_stack([B,C])
        ids=cov[id_col].astype(str)
        G=genotype.copy();G.index=G.index.astype(str)
        pid_order=pd.Index(pd.unique(ids),dtype=str)
        Gpart=G.reindex(pid_order)
        pid_to_row={p:i for i,p in enumerate(pid_order)}
        obs_to_part=np.fromiter((pid_to_row[x] for x in ids),dtype=int,count=len(ids))
        out=[]
        by_gene=cis_map.groupby("geneid")["snpid"].apply(list)
        for gene,snps in by_gene.items():
            if gene not in expression:continue
            ss=[s for s in snps if s in G]
            if not ss:continue
            y=expression[gene].to_numpy(float)
            # Gene-level nuisance covariance: fit once under the non-genetic model.
            bn=np.linalg.lstsq(X0,y,rcond=None)[0];r=y-X0@bn
            cm=fit_smooth_ct_covariance(age,r,groups,variance_df=self.variance_df,grouped_cache=grouped_cache)
            yw,(Xw,Bw),logdet=whiten_grouped(y,[X0,B],age,groups,cm,grouped_cache=grouped_cache)
            yt=torch.as_tensor(yw,dtype=self.dtype,device=self.device)
            xt=torch.as_tensor(Xw,dtype=self.dtype,device=self.device)
            bwt=torch.as_tensor(Bw,dtype=self.dtype,device=self.device)
            for st in range(0,len(ss),self.batch_size):
                s2=ss[st:st+self.batch_size]
                go_part=Gpart[s2].to_numpy(float).T
                go=go_part[:,obs_to_part]
                ch=genotype_channels(go,self.genotype_mode);nch=ch.shape[2];k=B.shape[1]
                Z=(ch[:,:,:,None]*Bw[None,:,None,:]).reshape(len(s2),len(y),nch*k)
                zt=torch.as_tensor(Z,dtype=self.dtype,device=self.device)
                rss0,rssf,bz,_=fwl_batch(yt,xt,zt)
                rss0n=float(rss0.cpu());rssfn=rssf.detach().cpu().numpy()
                df1=nch*k;df2=max(1,len(y)-X0.shape[1]-df1)
                F=((rss0n-rssfn)/df1)/(rssfn/df2)
                p=fdist.sf(np.maximum(F,0),df1,df2)
                main=[c*k for c in range(nch)]
                _,rssm,_,_=fwl_batch(yt,xt,zt[:,:,main]);rssmn=rssm.detach().cpu().numpy()
                ddf=nch*(k-1)
                Fd=((rssmn-rssfn)/ddf)/(rssfn/df2)
                dp=fdist.sf(np.maximum(Fd,0),ddf,df2)
                pr2=np.maximum(0,(rss0n-rssfn)/max(rss0n,1e-30))
                coef=bz.detach().cpu().numpy()
                for i,s in enumerate(s2):
                    row={"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(p[i]),
                         "Interaction_Pvalue":float(dp[i]),"F_total":float(F[i]),
                         "F_dynamic":float(Fd[i]),"partial_R2":float(pr2[i]),
                         "phi":float(cm.phi),"n_obs":len(y),"n_participants":len(groups),
                         "method":f"proposed_age_{self.basis_kind}_ctar_smoothvar",
                         "time_axis":"chronological_age","basis_kind":self.basis_kind,
                         "basis_json":json.dumps(basis.to_dict()),
                         "covariance_json":json.dumps(cm.to_dict()),
                         "age_min":basis.lower,"age_max":basis.upper}
                    for j,v in enumerate(coef[i]):row[f"beta_g_{j}"]=float(v)
                    out.append(row)
        z=pd.DataFrame(out)
        if len(z):
            q, iq, _ = stagewise_dynamic_fdr(z.Pvalue, z.Interaction_Pvalue, alpha=0.05)
            z["FDR"] = q; z["Interaction_FDR"] = iq
            z["runtime_sec_total"]=time.time()-t0
        return z
