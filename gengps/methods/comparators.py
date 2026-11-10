from __future__ import annotations
import time
import numpy as np,pandas as pd,torch
from scipy.stats import f as fdist, chi2
from ..basis import TimeBasis
from ..preprocess import visit_grid
from ..utils import bh_fdr,genotype_channels,require_aligned,stagewise_dynamic_fdr
from ..gpu import get_device,fwl_batch,configure_cuda_numerics

def _pointwise_batch(y,X0,gpart,genotype_mode,device,dtype):
    nch=1 if genotype_mode=="dosage" else 2
    chans=genotype_channels(gpart,genotype_mode)
    Z=chans # batch x n x c
    yt=torch.as_tensor(y,dtype=dtype,device=device)
    xt=torch.as_tensor(X0,dtype=dtype,device=device)
    zt=torch.as_tensor(Z,dtype=dtype,device=device)
    rss0,rssf,_,_=fwl_batch(yt,xt,zt)
    df1=nch;df2=max(1,len(y)-X0.shape[1]-nch)
    F=((float(rss0.cpu())-rssf.detach().cpu().numpy())/df1)/(rssf.detach().cpu().numpy()/df2)
    return np.asarray(fdist.sf(np.maximum(F,0),df1,df2),float)

class PolynomialRegressionScanner:
    """Lin et al. regression comparator; IID residuals, irregular rows allowed."""
    def __init__(self,basis_kind="cubic",genotype_mode="dummies",batch_size=4096,
                 device=None,dtype="float64",covariates=False):
        self.basis_kind=basis_kind;self.genotype_mode=genotype_mode;self.batch_size=batch_size
        self.device=get_device(device);configure_cuda_numerics(self.device,allow_tf32=False);self.dtype=torch.float64 if dtype=="float64" else torch.float32
        self.use_covariates=covariates
    def scan(self,cov,expression,genotype,cis_map,covariate_cols=None,
             id_col="participant_id",time_col="followup_time"):
        t0=time.time();require_aligned(cov,expression);covariate_cols=covariate_cols or []
        tb=TimeBasis(self.basis_kind,df=5).fit(cov[time_col]);B=tb.transform(cov[time_col])
        C=cov[covariate_cols].to_numpy(float) if (self.use_covariates and covariate_cols) else np.empty((len(cov),0))
        X0=np.column_stack([B,C]);ids=cov[id_col].astype(str)
        G=genotype.copy();G.index=G.index.astype(str)
        pid_order=pd.Index(pd.unique(ids),dtype=str)
        Gpart=G.reindex(pid_order)
        pid_to_row={p:i for i,p in enumerate(pid_order)}
        obs_to_part=np.fromiter((pid_to_row[x] for x in ids),dtype=int,count=len(ids))
        out=[]
        for gene,snps in cis_map.groupby("geneid")["snpid"].apply(list).items():
            if gene not in expression:continue
            ss=[s for s in snps if s in G]
            if not ss:continue
            y=expression[gene].to_numpy(float)
            yt=torch.as_tensor(y,dtype=self.dtype,device=self.device);xt=torch.as_tensor(X0,dtype=self.dtype,device=self.device)
            for st in range(0,len(ss),self.batch_size):
                s2=ss[st:st+self.batch_size]
                go_part=Gpart[s2].to_numpy(float).T
                go=go_part[:,obs_to_part]
                ch=genotype_channels(go,self.genotype_mode);nch=ch.shape[2];k=B.shape[1]
                Z=(ch[:,:,:,None]*B[None,:,None,:]).reshape(len(s2),len(y),nch*k)
                zt=torch.as_tensor(Z,dtype=self.dtype,device=self.device)
                rss0,rssf,_,_=fwl_batch(yt,xt,zt)
                df1=nch*k;df2=max(1,len(y)-X0.shape[1]-df1)
                F=((float(rss0.cpu())-rssf.cpu().numpy())/df1)/(rssf.cpu().numpy()/df2)
                p=fdist.sf(np.maximum(F,0),df1,df2)
                main=[c*k for c in range(nch)]
                _,rssm,_,_=fwl_batch(yt,xt,zt[:,:,main])
                ddf=nch*(k-1)
                Fd=((rssm.cpu().numpy()-rssf.cpu().numpy())/ddf)/(rssf.cpu().numpy()/df2)
                dp=fdist.sf(np.maximum(Fd,0),ddf,df2)
                for i,s in enumerate(s2):
                    out.append({"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(p[i]),
                                "Interaction_Pvalue":float(dp[i]),"F":float(F[i]),"F_dynamic":float(Fd[i]),
                                "method":f"{'cov_' if self.use_covariates else ''}{self.basis_kind}_regression",
                                "time_axis":time_col,"n_obs":len(y)})
        z=pd.DataFrame(out)
        if len(z):
            q, iq, _ = stagewise_dynamic_fdr(z.Pvalue, z.Interaction_Pvalue, alpha=0.05)
            z["FDR"] = q; z["Interaction_FDR"] = iq; z["runtime_sec_total"]=time.time()-t0
        return z

class UnionFisherScanner:
    """Pointwise association at each scheduled visit; missing visits are allowed."""
    def __init__(self,kind="union",genotype_mode="dummies",batch_size=4096,device=None,dtype="float64"):
        if kind not in {"union","fisher"}:raise ValueError
        self.kind=kind;self.genotype_mode=genotype_mode;self.batch_size=batch_size
        self.device=get_device(device);configure_cuda_numerics(self.device,allow_tf32=False);self.dtype=torch.float64 if dtype=="float64" else torch.float32
    def scan(self,cov,expression,genotype,cis_map,id_col="participant_id",visit_col="visit"):
        t0=time.time();require_aligned(cov,expression);G=genotype.copy();G.index=G.index.astype(str)
        visits=list(pd.unique(cov[visit_col].dropna()))
        out=[]
        for gene,snps in cis_map.groupby("geneid")["snpid"].apply(list).items():
            if gene not in expression:continue
            ss=[s for s in snps if s in G]
            if not ss:continue
            for st in range(0,len(ss),self.batch_size):
                s2=ss[st:st+self.batch_size];p_by=[]
                for v in visits:
                    mask=(cov[visit_col].values==v) & np.isfinite(expression[gene].values)
                    if mask.sum()<10:continue
                    y=expression.loc[mask,gene].to_numpy(float)
                    ids=cov.loc[mask,id_col].astype(str)
                    gp=G.reindex(ids)[s2].to_numpy(float).T
                    p=_pointwise_batch(y,np.ones((len(y),1)),gp,self.genotype_mode,self.device,self.dtype)
                    p_by.append(p)
                if not p_by:continue
                P=np.vstack(p_by)
                if self.kind=="union":
                    stat=np.nanmin(P,axis=0)
                    # Analytical Bonferroni is supplied for comparability; Lin et al. calibrated by permutation.
                    pv=np.minimum(1,stat*P.shape[0])
                else:
                    stat=-2*np.nansum(np.log(np.clip(P,1e-300,1)),axis=0)
                    pv=chi2.sf(stat,2*P.shape[0])
                for i,s in enumerate(s2):
                    out.append({"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(pv[i]),
                                "statistic":float(stat[i]),"n_timepoints_used":P.shape[0],
                                "method":self.kind,"time_axis":"visit"})
        z=pd.DataFrame(out)
        if len(z):z["FDR"]=bh_fdr(z.Pvalue);z["runtime_sec_total"]=time.time()-t0
        return z

class AR1AssociationScanner:
    """Lin et al. AR comparator using adjacent observed visits within participants."""
    def __init__(self,genotype_mode="dummies",batch_size=4096,device=None,dtype="float64"):
        self.genotype_mode=genotype_mode;self.batch_size=batch_size;self.device=get_device(device);configure_cuda_numerics(self.device,allow_tf32=False)
        self.dtype=torch.float64 if dtype=="float64" else torch.float32
    def scan(self,cov,expression,genotype,cis_map,id_col="participant_id",time_col="followup_time"):
        t0=time.time();require_aligned(cov,expression);G=genotype.copy();G.index=G.index.astype(str)
        pairs=[]
        for pid,g in cov.groupby(id_col,sort=False):
            pos=np.array([cov.index.get_loc(i) for i in g.sort_values(time_col).index])
            if len(pos)>1:pairs.extend(zip(pos[:-1],pos[1:]))
        if not pairs:return pd.DataFrame()
        prev=np.array([a for a,b in pairs]);now=np.array([b for a,b in pairs])
        ids=cov.iloc[now][id_col].astype(str)
        out=[]
        for gene,snps in cis_map.groupby("geneid")["snpid"].apply(list).items():
            if gene not in expression:continue
            ss=[s for s in snps if s in G]
            if not ss:continue
            yp=expression.iloc[prev][gene].to_numpy(float);yn=expression.iloc[now][gene].to_numpy(float)
            ok=np.isfinite(yp)&np.isfinite(yn);yp=yp[ok];yn=yn[ok];ids2=ids.iloc[np.flatnonzero(ok)]
            X0=np.column_stack([np.ones(len(yp)),yp])
            yt=torch.as_tensor(yn,dtype=self.dtype,device=self.device);xt=torch.as_tensor(X0,dtype=self.dtype,device=self.device)
            for st in range(0,len(ss),self.batch_size):
                s2=ss[st:st+self.batch_size];gp=G.reindex(ids2)[s2].to_numpy(float).T
                ch=genotype_channels(gp,self.genotype_mode);nch=ch.shape[2]
                # genotype-specific intercept and AR slope
                Z=np.concatenate([ch,ch*yp[None,:,None]],axis=2)
                zt=torch.as_tensor(Z,dtype=self.dtype,device=self.device)
                rss0,rssf,_,_=fwl_batch(yt,xt,zt)
                df1=2*nch;df2=max(1,len(yp)-X0.shape[1]-df1)
                F=((float(rss0.cpu())-rssf.cpu().numpy())/df1)/(rssf.cpu().numpy()/df2)
                p=fdist.sf(np.maximum(F,0),df1,df2)
                for i,s in enumerate(s2):
                    out.append({"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(p[i]),"F":float(F[i]),
                                "n_pairs":len(yp),"method":"lin2018_AR1","time_axis":time_col})
        z=pd.DataFrame(out)
        if len(z):z["FDR"]=bh_fdr(z.Pvalue);z["runtime_sec_total"]=time.time()-t0
        return z

class MANOVAScanner:
    """Fast Wilks-lambda MANOVA with Bartlett chi-square approximation.

    MANOVA inherently needs a complete multivariate response vector for the
    scheduled visits, so complete participants are used for this comparator.
    """
    def __init__(self,genotype_mode="dummies"):
        self.genotype_mode=genotype_mode
    def scan(self,cov,expression,genotype,cis_map,id_col="participant_id",visit_col="visit"):
        t0=time.time();require_aligned(cov,expression);G=genotype.copy();G.index=G.index.astype(str)
        pids,visits,lookup=visit_grid(cov,visit_col,id_col)
        complete=np.all(lookup>=0,axis=1)
        pids2=np.array(pids)[complete];lk=lookup[complete]
        out=[]
        for gene,snps in cis_map.groupby("geneid")["snpid"].apply(list).items():
            if gene not in expression:continue
            Y=np.array([[expression.iloc[r][gene] for r in row] for row in lk],float)
            ok=np.all(np.isfinite(Y),axis=1);Y=Y[ok];ids=pids2[ok]
            if len(Y)<max(8,len(visits)+4):continue
            Yc=Y-Y.mean(axis=0,keepdims=True);E0=Yc.T@Yc
            det0=max(np.linalg.det(E0),1e-300)
            for s in [x for x in snps if x in G]:
                g=G.reindex(ids)[s].to_numpy(float)
                ch=genotype_channels(g,self.genotype_mode).reshape(len(g),-1)
                X=np.column_stack([np.ones(len(g)),ch])
                B=np.linalg.lstsq(X,Y,rcond=None)[0];R=Y-X@B;E=R.T@R
                lam=max(min(np.linalg.det(E)/det0,1.0),1e-300)
                p_dim=Y.shape[1];q=ch.shape[1];n=len(Y)
                bart=-(n-1-(p_dim+q+1)/2)*np.log(lam)
                pv=chi2.sf(max(bart,0),p_dim*q)
                out.append({"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(pv),
                            "Wilks_lambda":float(lam),"n_participants":n,
                            "method":"MANOVA_Wilks_Bartlett","time_axis":"visit"})
        z=pd.DataFrame(out)
        if len(z):z["FDR"]=bh_fdr(z.Pvalue);z["runtime_sec_total"]=time.time()-t0
        return z
