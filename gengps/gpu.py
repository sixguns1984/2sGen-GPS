from __future__ import annotations
import torch


def get_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def safe_solve(A, b, ridge=1e-10):
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A + ridge * I, b)


def fwl_batch(y, X0, Z):
    """Batched Frisch-Waugh-Lovell with one response shared by all tests.

    Parameters
    ----------
    y : (n,)
    X0 : (n, p)
    Z : (batch, n, k)

    Returns
    -------
    rss0 : scalar tensor
    rssfull : (batch,)
    beta_z : (batch, k)
    Zr : (batch, n, k)
    """
    inv = torch.linalg.pinv(X0.T @ X0)
    by = inv @ (X0.T @ y)
    yr = y - X0 @ by
    XtZ = torch.einsum("np,bnk->bpk", X0, Z)
    cx = torch.einsum("pq,bqk->bpk", inv, XtZ)
    Zr = Z - torch.einsum("np,bpk->bnk", X0, cx)
    ztz = torch.einsum("bnk,bnl->bkl", Zr, Zr)
    zty = torch.einsum("bnk,n->bk", Zr, yr)
    bz = safe_solve(ztz, zty.unsqueeze(-1)).squeeze(-1)
    fit = torch.einsum("bnk,bk->bn", Zr, bz)
    rssf = torch.sum((yr[None, :] - fit) ** 2, dim=1)
    rss0 = yr @ yr
    return rss0, rssf, bz, Zr


def fwl_batch_y(y, X0, Z):
    """Batched FWL when every test has its own response vector.

    This is used by the paper-style Lin2018 missing-data workflow: missing
    scheduled observations are imputed from each SNP's genotype-specific
    polynomial curves, so the completed response differs by SNP.

    Parameters
    ----------
    y : (batch, n)
    X0 : (n, p)
    Z : (batch, n, k)

    Returns
    -------
    rss0 : (batch,)
    rssfull : (batch,)
    beta_z : (batch, k)
    Zr : (batch, n, k)
    """
    inv = torch.linalg.pinv(X0.T @ X0)
    Xty = torch.einsum("np,bn->bp", X0, y)
    by = torch.einsum("pq,bq->bp", inv, Xty)
    yr = y - torch.einsum("np,bp->bn", X0, by)

    XtZ = torch.einsum("np,bnk->bpk", X0, Z)
    cx = torch.einsum("pq,bqk->bpk", inv, XtZ)
    Zr = Z - torch.einsum("np,bpk->bnk", X0, cx)

    ztz = torch.einsum("bnk,bnl->bkl", Zr, Zr)
    zty = torch.einsum("bnk,bn->bk", Zr, yr)
    bz = safe_solve(ztz, zty.unsqueeze(-1)).squeeze(-1)
    fit = torch.einsum("bnk,bk->bn", Zr, bz)
    rssf = torch.sum((yr - fit) ** 2, dim=1)
    rss0 = torch.sum(yr ** 2, dim=1)
    return rss0, rssf, bz, Zr


def full_ols_batch_shared_y(y, X0, Z):
    """Fit [X0, Z_b] to a response shared across a test batch.

    Returns the X0 and Z coefficients for every member of the batch.  It is
    algebraically the same OLS fit as constructing a separate full design
    matrix for every SNP, but avoids that large concatenated matrix.
    """
    inv = torch.linalg.pinv(X0.T @ X0)
    by0 = inv @ (X0.T @ y)
    yr = y - X0 @ by0

    XtZ = torch.einsum("np,bnk->bpk", X0, Z)
    cx = torch.einsum("pq,bqk->bpk", inv, XtZ)
    Zr = Z - torch.einsum("np,bpk->bnk", X0, cx)

    ztz = torch.einsum("bnk,bnl->bkl", Zr, Zr)
    zty = torch.einsum("bnk,n->bk", Zr, yr)
    bz = safe_solve(ztz, zty.unsqueeze(-1)).squeeze(-1)

    # Conditional X0 coefficient after accounting for the fitted genetic block.
    resid_for_x = y[None, :] - torch.einsum("bnk,bk->bn", Z, bz)
    Xty = torch.einsum("np,bn->bp", X0, resid_for_x)
    bx = torch.einsum("pq,bq->bp", inv, Xty)
    return bx, bz


def configure_cuda_numerics(device=None, allow_tf32=False):
    """Configure CUDA math for reproducible statistical inference.

    Production scans use FP64 in the current v8.9 workflow. TF32 remains
    disabled so Ampere devices never silently lower matrix-multiply precision.
    """
    dev = get_device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    return dev
