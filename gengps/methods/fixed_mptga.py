from __future__ import annotations
import json, time
import numpy as np, pandas as pd, torch
from scipy.stats import chi2
from ..basis import TimeBasis
from ..preprocess import filter_complete_grid
from ..utils import bh_fdr, genotype_channels, stagewise_dynamic_fdr
from ..gpu import get_device, safe_solve, fwl_batch, configure_cuda_numerics


def _profile_ll(rss, nobs, logdet_total):
    rss=np.maximum(np.asarray(rss,float),1e-30)
    return -.5*(nobs*np.log(2*np.pi)+logdet_total+nobs*np.log(rss/nobs)+nobs)


def _profile_ll_torch(rss, nobs, logdet_total):
    rss=torch.clamp(rss,min=1e-30)
    const=float(nobs*np.log(2*np.pi)+logdet_total+nobs)
    return -0.5*(const + nobs*torch.log(rss/float(nobs)))


def _cross_components_global(U, V):
    """Three sufficient-statistic components for U' Q(rho) V.

    U: (N,T,p), V: (N,T,q).  For an AR(1) correlation matrix,
    Q(rho)=R(rho)^-1 is tridiagonal and can be written as a linear
    combination of three fixed cross-product components with weights
    [1/(1-rho^2), -rho/(1-rho^2), (1+rho^2)/(1-rho^2)].

    This is the same decomposition used by the original teQTL0.py alpha()/ymat()
    implementation, but here it is combined with an FWL/Schur-complement GLS
    calculation so the large nuisance block is never repeated per SNP.
    """
    ep = torch.einsum("ntp,ntq->pq", U[:, [0, -1], :], V[:, [0, -1], :])
    adj = (
        torch.einsum("ntp,ntq->pq", U[:, :-1, :], V[:, 1:, :])
        + torch.einsum("ntp,ntq->pq", U[:, 1:, :], V[:, :-1, :])
    )
    if U.shape[1] > 2:
        interior = torch.einsum("ntp,ntq->pq", U[:, 1:-1, :], V[:, 1:-1, :])
    else:
        interior = torch.zeros_like(ep)
    return torch.stack([ep, adj, interior], dim=0)


def _cross_components_by_participant(U, V):
    """Participant-specific analogue of _cross_components_global.

    Returns (3,N,p,q) without summing over participants.
    """
    ep = torch.einsum("ntp,ntq->npq", U[:, [0, -1], :], V[:, [0, -1], :])
    adj = (
        torch.einsum("ntp,ntq->npq", U[:, :-1, :], V[:, 1:, :])
        + torch.einsum("ntp,ntq->npq", U[:, 1:, :], V[:, :-1, :])
    )
    if U.shape[1] > 2:
        interior = torch.einsum("ntp,ntq->npq", U[:, 1:-1, :], V[:, 1:-1, :])
    else:
        interior = torch.zeros_like(ep)
    return torch.stack([ep, adj, interior], dim=0)


def _yy_components(Y):
    """AR(1) three-component sufficient statistic for y'Qy, all genes.

    Y: (N,T,G), returns (3,G).
    """
    ep = torch.sum(Y[:, [0, -1], :] * Y[:, [0, -1], :], dim=(0,1))
    adj = 2.0 * torch.sum(Y[:, :-1, :] * Y[:, 1:, :], dim=(0,1))
    if Y.shape[1] > 2:
        interior = torch.sum(Y[:, 1:-1, :] * Y[:, 1:-1, :], dim=(0,1))
    else:
        interior = torch.zeros_like(ep)
    return torch.stack([ep, adj, interior], dim=0)


def _ar1_weights(rho, *, dtype, device):
    den=1.0-float(rho)*float(rho)
    return torch.tensor(
        [1.0/den, -float(rho)/den, (1.0+float(rho)*float(rho))/den],
        dtype=dtype, device=device,
    )


class CompleteGridMPTGA:
    """Profile-likelihood MPTGA on a complete equal visit grid.

    Stage-I manuscript mode reproduces the submitted model:
      * common raw cubic time curve [1,t,t^2,t^3]
      * two legacy BinaryEncoder genotype channels
      * genotype x cubic-time genetic block
      * discrete AR(1) residual covariance
      * rho = 0.00,0.01,...,0.99

    v8.9-fast uses the AR(1) sufficient-statistic decomposition from the
    original hand-written teQTL0.py, combined with a Frisch-Waugh-Lovell
    Schur complement.  The statistical model and likelihood are unchanged,
    but it avoids constructing/transferring (SNP x N*T x genetic_terms)
    tensors for every rho and avoids repeating the nuisance pseudo-inverse.
    """
    def __init__(self,basis_kind="cubic",genotype_mode="dummies",rho_points=40,
                 batch_size=2048,device=None,dtype="float64",rho_grid_mode="linspace_098"):
        self.basis_kind=basis_kind;self.genotype_mode=genotype_mode
        self.rho_grid_mode=str(rho_grid_mode)
        if self.rho_grid_mode == "legacy_exact":
            if int(rho_points) != 100:
                raise ValueError("rho_grid_mode='legacy_exact' requires rho_points=100")
            self.rho_grid=np.arange(100,dtype=float)/100.0
        elif self.rho_grid_mode == "linspace_098":
            self.rho_grid=np.linspace(0,.98,int(rho_points))
        else:
            raise ValueError(f"Unknown rho_grid_mode: {rho_grid_mode}")
        self.batch_size=int(batch_size)
        self.device=get_device(device)
        configure_cuda_numerics(self.device,allow_tf32=False)
        self.dtype=torch.float64 if dtype=="float64" else torch.float32
        self.ar_jitter = 0.0 if self.rho_grid_mode == "legacy_exact" else 1e-8

    def _scan_sufficient_stats(self, cov, expression, genotype, cis_map,
                               covariate_cols, id_col, visit_col, time_col,
                               required_visits):
        t0=time.time()
        cov,expression,genotype,visits=filter_complete_grid(
            cov,expression,genotype,visit_col,id_col,required_visits)
        n_part=genotype.shape[0]; n_visit=len(visits)
        vt=cov.groupby(visit_col)[time_col].median().reindex(visits).to_numpy(float)
        basis=TimeBasis(self.basis_kind,df=5).fit(vt)
        if self.rho_grid_mode == "legacy_exact" and self.basis_kind == "cubic":
            Bv=np.column_stack([np.ones_like(vt), vt, vt*vt, vt*vt*vt])
            basis_parameterization="raw_time_polynomial"
        else:
            Bv=basis.transform(vt)
            basis_parameterization="standardized_time_basis"

        B=np.tile(Bv,(n_part,1))
        C=cov[covariate_cols].to_numpy(float) if covariate_cols else np.empty((len(cov),0))
        X0=np.column_stack([B,C])
        genotype=genotype.copy(); genotype.index=genotype.index.astype(str)
        by_gene=cis_map.groupby("geneid", sort=False)["snpid"].apply(list)
        genes_use=[g for g in by_gene.index.astype(str) if g in expression.columns]
        if not genes_use:
            return pd.DataFrame()
        gene_pos={g:i for i,g in enumerate(genes_use)}

        # Transfer the complete-grid nuisance design and all chunk genes once.
        X3=torch.as_tensor(
            X0.reshape(n_part,n_visit,X0.shape[1]), dtype=self.dtype, device=self.device)
        Y3=torch.as_tensor(
            expression[genes_use].to_numpy(float).reshape(n_part,n_visit,len(genes_use)),
            dtype=self.dtype, device=self.device)
        B3=torch.as_tensor(Bv[None,:,:],dtype=self.dtype,device=self.device)

        # The three fixed AR(1) sufficient-statistic components.  These are the
        # key acceleration borrowed from the original teQTL0.py alpha()/ymat().
        Acomp=_cross_components_global(X3,X3)                 # 3,p,p
        Hcomp=_cross_components_by_participant(X3,B3.expand(n_part,-1,-1)) # 3,N,p,k
        Kcomp=_cross_components_global(B3,B3)                 # 3,k,k (one participant)
        acomp=_cross_components_global(X3,Y3)                 # 3,p,G
        Jcomp=_cross_components_by_participant(B3.expand(n_part,-1,-1),Y3) # 3,N,k,G
        yycomp=_yy_components(Y3)                             # 3,G

        nobs=n_part*n_visit
        p=X0.shape[1]; k=Bv.shape[1]; G=len(genes_use)

        # Cache only small rho-dependent matrices.  No SNP x N*T tensors are
        # created here.  Ainv is computed once per rho per chunk, not per SNP batch.
        rho_cache=[]
        for rho in self.rho_grid:
            w=_ar1_weights(rho,dtype=self.dtype,device=self.device)
            A=torch.einsum("h,hpq->pq",w,Acomp)
            Ainv=torch.linalg.pinv(A)
            a=torch.einsum("h,hpg->pg",w,acomp)
            u=Ainv@a
            yy=torch.einsum("h,hg->g",w,yycomp)
            rss0=yy-torch.sum(a*u,dim=0)
            rss0=torch.clamp(rss0,min=1e-30)
            if self.rho_grid_mode == "legacy_exact":
                logdet=float(n_part*(n_visit-1)*np.log(1.0-float(rho)*float(rho)))
            else:
                # For the non-manuscript jittered path use the exact small R determinant.
                d=np.abs(np.arange(n_visit)[:,None]-np.arange(n_visit)[None,:])
                R=np.power(float(rho),d)
                if self.ar_jitter:
                    R.flat[::n_visit+1]+=float(self.ar_jitter)
                sign,ld=np.linalg.slogdet(R)
                if sign <= 0:
                    raise np.linalg.LinAlgError("AR(1) covariance is not positive definite")
                logdet=float(n_part*ld)
            ll0=_profile_ll_torch(rss0,nobs,logdet)
            Kr=torch.einsum("h,hkl->kl",w,Kcomp)
            rho_cache.append((float(rho),w,Ainv,u,rss0,ll0,Kr,logdet))

        out=[]
        for gene,snps in by_gene.items():
            gene=str(gene)
            if gene not in gene_pos:
                continue
            snps=[s for s in snps if s in genotype.columns]
            if not snps:
                continue
            gi=gene_pos[gene]
            # Profiled nuisance-only likelihood for this gene.
            best_null=max(float(x[5][gi].detach().cpu()) for x in rho_cache)

            Gp=genotype[snps]
            for st in range(0,len(snps),self.batch_size):
                ss=snps[st:st+self.batch_size]
                gpart=Gp[ss].to_numpy(float).T
                chans_np=genotype_channels(gpart,self.genotype_mode)
                gt=torch.as_tensor(chans_np,dtype=self.dtype,device=self.device)
                bsz,_,nch=gt.shape
                q=nch*k

                # Genotype-only participant sums, and the three weighted cross-product
                # components.  They are computed ONCE per SNP batch, then reused for
                # all 100 rho values.
                Sg=torch.einsum("bic,bid->bcd",gt,gt)  # b,c,c
                Bcomp=torch.einsum("bic,hipk->hbpck",gt,Hcomp)  # 3,b,p,c,k
                Dcomp=torch.einsum("bic,hik->hbck",gt,Jcomp[:,:,:,gi]) # 3,b,c,k

                best_full=torch.full((bsz,),-torch.inf,dtype=self.dtype,device=self.device)
                best_main=torch.full((bsz,),-torch.inf,dtype=self.dtype,device=self.device)
                best_rho=torch.full((bsz,),float("nan"),dtype=self.dtype,device=self.device)
                best_beta=torch.full((bsz,q),float("nan"),dtype=self.dtype,device=self.device)
                main_idx=torch.as_tensor([c*k for c in range(nch)],dtype=torch.long,device=self.device)

                for rho,w,Ainv,u_all,rss0_all,ll0_all,Kr,logdet in rho_cache:
                    # X'QZ and Z'Qy from the three legacy sufficient-statistic components.
                    XB=torch.einsum("h,hbpck->bpck",w,Bcomp).reshape(bsz,p,q)
                    dvec=torch.einsum("h,hbck->bck",w,Dcomp).reshape(bsz,q)

                    # Z'QZ has a Kronecker structure because genotype is participant-
                    # constant and the genetic time basis is common to all participants.
                    Cfull=torch.einsum("bcd,kl->bckdl",Sg,Kr).reshape(bsz,q,q)

                    # Weighted FWL/Schur complement:
                    # S = Z' M_X^Q Z ; t = Z' M_X^Q y.
                    AinvB=torch.einsum("pq,bqk->bpk",Ainv,XB)
                    S=Cfull-torch.einsum("bpk,bpl->bkl",XB,AinvB)
                    tvec=dvec-torch.einsum("bpk,p->bk",XB,u_all[:,gi])

                    beta=safe_solve(S,tvec.unsqueeze(-1)).squeeze(-1)
                    # safe_solve has a tiny ridge, but the reported likelihood is the
                    # unpenalized GLS likelihood, matching the previous fwl_batch path.
                    rss0=rss0_all[gi]
                    rssf=(
                        rss0
                        -2.0*torch.sum(beta*tvec,dim=1)
                        +torch.einsum("bi,bij,bj->b",beta,S,beta)
                    )
                    llf=_profile_ll_torch(rssf,nobs,logdet)
                    take=llf>best_full
                    best_full=torch.where(take,llf,best_full)
                    best_rho=torch.where(take,torch.full_like(best_rho,rho),best_rho)
                    best_beta=torch.where(take[:,None],beta,best_beta)

                    Sm=S.index_select(1,main_idx).index_select(2,main_idx)
                    tm=tvec.index_select(1,main_idx)
                    bm=safe_solve(Sm,tm.unsqueeze(-1)).squeeze(-1)
                    rssm=(
                        rss0
                        -2.0*torch.sum(bm*tm,dim=1)
                        +torch.einsum("bi,bij,bj->b",bm,Sm,bm)
                    )
                    llm=_profile_ll_torch(rssm,nobs,logdet)
                    best_main=torch.maximum(best_main,llm)

                best_full_np=best_full.detach().cpu().numpy()
                best_main_np=best_main.detach().cpu().numpy()
                best_rho_np=best_rho.detach().cpu().numpy()
                best_beta_np=best_beta.detach().cpu().numpy()
                lrt=2.0*(best_full_np-best_null)
                dlrt=2.0*(best_full_np-best_main_np)
                pval=chi2.sf(np.maximum(lrt,0),nch*k)
                dp=chi2.sf(np.maximum(dlrt,0),nch*(k-1))

                for i,s in enumerate(ss):
                    row={
                        "Variant_ID":s,"Ensembl_ID":gene,
                        "Pvalue":float(pval[i]),
                        "Interaction_Pvalue":float(dp[i]),
                        "LRT":float(lrt[i]),"Dynamic_LRT":float(dlrt[i]),
                        "rho":float(best_rho_np[i]),
                        "n_obs":nobs,"n_participants":n_part,"n_visits":n_visit,
                        "basis_kind":self.basis_kind,"time_axis":"followup_time",
                        "basis_parameterization":basis_parameterization,
                        "method":"submitted_followup_mptga_fast",
                        "rho_grid_mode":self.rho_grid_mode,
                        "rho_grid_points":int(len(self.rho_grid)),
                        "compute_engine":"ar1_sufficient_stats_fwl_v1",
                        "basis_json":json.dumps(basis.to_dict()),
                    }
                    for j,v in enumerate(best_beta_np[i]):
                        row[f"beta_g_{j}"]=float(v)
                    out.append(row)

        z=pd.DataFrame(out)
        if len(z):
            q, iq, _ = stagewise_dynamic_fdr(z.Pvalue, z.Interaction_Pvalue, alpha=0.05)
            z["FDR"] = q
            z["Interaction_FDR"] = iq
            z["runtime_sec_total"]=time.time()-t0
        return z

    def _scan_whiten_reference(self,cov,expression,genotype,cis_map,covariate_cols,
                               id_col,visit_col,time_col,required_visits):
        """Original v8.9 whitening/FWL engine retained for non-manuscript modes."""
        t0=time.time()
        cov,expression,genotype,visits=filter_complete_grid(
            cov,expression,genotype,visit_col,id_col,required_visits)
        n_part=genotype.shape[0];n_visit=len(visits)
        vt=cov.groupby(visit_col)[time_col].median().reindex(visits).to_numpy(float)
        basis=TimeBasis(self.basis_kind,df=5).fit(vt)
        Bv=basis.transform(vt)
        basis_parameterization="standardized_time_basis"
        B=np.tile(Bv,(n_part,1))
        C=cov[covariate_cols].to_numpy(float) if covariate_cols else np.empty((len(cov),0))
        X0=np.column_stack([B,C])
        genotype=genotype.copy();genotype.index=genotype.index.astype(str)
        by_gene=cis_map.groupby("geneid", sort=False)["snpid"].apply(list)
        out=[]
        genes_use=[g for g in by_gene.index.astype(str) if g in expression.columns]
        gene_pos={g:i for i,g in enumerate(genes_use)}
        Y=expression[genes_use].to_numpy(float) if genes_use else np.empty((len(cov),0),float)
        Y3=Y.reshape(n_part,n_visit,len(genes_use))
        XX=X0.reshape(n_part,n_visit,X0.shape[1])
        d=np.abs(np.arange(n_visit)[:,None]-np.arange(n_visit)[None,:])
        rho_cache=[]
        for rho in self.rho_grid:
            R=np.power(float(rho),d)
            if float(self.ar_jitter) != 0.0:
                R.flat[::n_visit+1]+=float(self.ar_jitter)
            L=np.linalg.cholesky(R)
            Xw=np.linalg.solve(L,XX).reshape(-1,X0.shape[1])
            Bw=np.linalg.solve(L,Bv)
            if len(genes_use):
                rhs=Y3.transpose(1,0,2).reshape(n_visit,-1)
                Yw=np.linalg.solve(L,rhs).reshape(n_visit,n_part,len(genes_use)).transpose(1,0,2).reshape(-1,len(genes_use))
            else:
                Yw=np.empty((n_part*n_visit,0),float)
            logdet=float(n_part*2*np.log(np.diag(L)).sum())
            xt=torch.as_tensor(Xw,dtype=self.dtype,device=self.device)
            rho_cache.append((float(rho),Yw,xt,Bw,logdet))

        for gene,snps in by_gene.items():
            gene=str(gene)
            if gene not in gene_pos: continue
            snps=[s for s in snps if s in genotype.columns]
            if not snps: continue
            gi=gene_pos[gene]
            y=Y[:,gi]
            prof=[]
            for rho,Yw,xt,Bw,logdet in rho_cache:
                yw=Yw[:,gi]
                yt=torch.as_tensor(yw,dtype=self.dtype,device=self.device)
                b0=torch.linalg.lstsq(xt,yt).solution
                r0=yt-xt@b0
                rss0=float((r0@r0).cpu())
                ll0=float(_profile_ll(rss0,len(y),logdet))
                prof.append((rho,yw,xt,Bw,logdet,ll0))
            best_null=max(x[5] for x in prof)
            Gp=genotype[snps]
            for st in range(0,len(snps),self.batch_size):
                ss=snps[st:st+self.batch_size]
                gpart=Gp[ss].to_numpy(float).T
                chans=genotype_channels(gpart,self.genotype_mode)
                bsz,_,nch=chans.shape;k=Bv.shape[1]
                best_full=np.full(bsz,-np.inf);best_rho=np.full(bsz,np.nan)
                best_beta=np.full((bsz,nch*k),np.nan);best_main=np.full(bsz,-np.inf)
                for rho,yw,xt,Bw,logdet,ll0 in prof:
                    Z=chans[:,:,:,None,None]*Bw[None,None,None,:,:]
                    Z=np.transpose(Z,(0,1,3,2,4)).reshape(bsz,n_part*n_visit,nch*k)
                    yt=torch.as_tensor(yw,dtype=self.dtype,device=self.device)
                    zt=torch.as_tensor(Z,dtype=self.dtype,device=self.device)
                    rss0t,rssf,bz,Zr=fwl_batch(yt,xt,zt)
                    llf=_profile_ll(rssf.detach().cpu().numpy(),len(y),logdet)
                    take=llf>best_full
                    if np.any(take):
                        best_full[take]=llf[take];best_rho[take]=rho
                        best_beta[take]=bz.detach().cpu().numpy()[take]
                    main_cols=[c*k for c in range(nch)]
                    zm=zt[:,:,main_cols]
                    _,rssm,_,_=fwl_batch(yt,xt,zm)
                    llm=_profile_ll(rssm.detach().cpu().numpy(),len(y),logdet)
                    best_main=np.maximum(best_main,llm)
                lrt=2*(best_full-best_null);p=chi2.sf(np.maximum(lrt,0),nch*k)
                dlrt=2*(best_full-best_main);dp=chi2.sf(np.maximum(dlrt,0),nch*(k-1))
                for i,s in enumerate(ss):
                    row={"Variant_ID":s,"Ensembl_ID":gene,"Pvalue":float(p[i]),
                         "Interaction_Pvalue":float(dp[i]),"LRT":float(lrt[i]),
                         "Dynamic_LRT":float(dlrt[i]),"rho":float(best_rho[i]),
                         "n_obs":len(y),"n_participants":n_part,"n_visits":n_visit,
                         "basis_kind":self.basis_kind,"time_axis":"followup_time",
                         "basis_parameterization":basis_parameterization,
                         "method":"submitted_followup_mptga_fast",
                         "rho_grid_mode":self.rho_grid_mode,
                         "rho_grid_points":int(len(self.rho_grid)),
                         "compute_engine":"whiten_fwl_v89_reference",
                         "basis_json":json.dumps(basis.to_dict())}
                    for j,v in enumerate(best_beta[i]): row[f"beta_g_{j}"]=float(v)
                    out.append(row)
        z=pd.DataFrame(out)
        if len(z):
            q, iq, _ = stagewise_dynamic_fdr(z.Pvalue, z.Interaction_Pvalue, alpha=0.05)
            z["FDR"] = q; z["Interaction_FDR"] = iq
            z["runtime_sec_total"]=time.time()-t0
        return z

    def scan(self,cov,expression,genotype,cis_map,covariate_cols=None,
             id_col="participant_id",visit_col="visit",time_col="followup_time",
             required_visits=None):
        covariate_cols=covariate_cols or []
        # The sufficient-statistic engine is exact for the unjittered legacy AR(1)
        # covariance used by manuscript Stage-I.  Current-v8.9 exploratory modes
        # retain their original jittered whitening implementation.
        if self.rho_grid_mode == "legacy_exact":
            return self._scan_sufficient_stats(
                cov,expression,genotype,cis_map,covariate_cols,
                id_col,visit_col,time_col,required_visits)
        return self._scan_whiten_reference(
            cov,expression,genotype,cis_map,covariate_cols,
            id_col,visit_col,time_col,required_visits)
