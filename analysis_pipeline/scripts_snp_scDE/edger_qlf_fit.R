#!/usr/bin/env Rscript
# 2sGenGPS postmortem v6.3: edgeR 4.x QLF, raw donor-pseudobulk counts.
# Args: genes_x_donors.mtx genes.csv donors.csv result.csv min_count min_total min_prop robust01
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8L) {
  stop('Usage: Rscript edger_qlf_fit.R counts.mtx genes.csv donors.csv result.csv min_count min_total min_prop robust01')
}
mtx_path <- args[[1L]]
gene_path <- args[[2L]]
donor_path <- args[[3L]]
out_path <- args[[4L]]
min_count <- as.numeric(args[[5L]])
min_total <- as.numeric(args[[6L]])
min_prop <- as.numeric(args[[7L]])
use_robust <- as.integer(args[[8L]]) == 1L
stopifnot(is.finite(min_count), min_count > 0, is.finite(min_total), min_total > 0,
          is.finite(min_prop), min_prop > 0, min_prop <= 1)

suppressPackageStartupMessages({
  library(edgeR)
  library(Matrix)
})
if (utils::packageVersion('edgeR') < '4.0.0') {
  stop('This module targets edgeR >= 4.0.0 (tested interface: glmQLFit legacy=FALSE)')
}

md <- utils::read.csv(donor_path, stringsAsFactors=FALSE, check.names=FALSE)
genes <- utils::read.csv(gene_path, stringsAsFactors=FALSE, check.names=FALSE)
required <- c('donor','carrier','ageScaled','sex','status','n_cells')
if (!all(required %in% colnames(md)) || !('gene_id' %in% colnames(genes))) {
  stop('Required prepared donor / gene columns are missing')
}
if (anyNA(md[, required]) || anyDuplicated(md$donor) || anyDuplicated(genes$gene_id)) {
  stop('NA or duplicated donor/gene identifiers in prepared data')
}
if (!all(md$carrier %in% c(0L,1L)) || !all(md$sex %in% c('M','F')) ||
    !all(md$status %in% c('Case','Control')) || !all(is.finite(md$ageScaled)) ||
    !all(md$n_cells >= 1)) stop('Invalid donor metadata values')

# Input MatrixMarket is genes x donors (unaltered raw integer counts).
y_sparse <- as(Matrix::readMM(mtx_path), 'dgCMatrix')
if (nrow(y_sparse) != nrow(genes) || ncol(y_sparse) != nrow(md)) {
  stop('Dimension mismatch: counts genes x donors != metadata')
}
if (any(!is.finite(y_sparse@x)) || any(y_sparse@x < 0) ||
    any(abs(y_sparse@x - round(y_sparse@x)) > 1e-5)) {
  stop('edgeR requires nonnegative raw counts')
}
if (any(Matrix::colSums(y_sparse) <= 0)) stop('Zero pseudobulk library')
count_mat <- as.matrix(y_sparse)
rownames(count_mat) <- as.character(genes$gene_id)
colnames(count_mat) <- as.character(md$donor)
# Free sparse copy before GLM fitting.
rm(y_sparse)

md$carrier <- factor(as.character(md$carrier), levels=c('0','1'))
md$sex <- factor(md$sex,levels=c('F','M'))
md$status <- factor(md$status,levels=c('Control','Case'))
md$ageScaled <- as.numeric(md$ageScaled)
rownames(md) <- md$donor

# Same covariate inclusion rules as postmortem_scde.design_for.
terms <- c('carrier', 'ageScaled')
if (length(unique(md$sex)) > 1L) terms <- c(terms, 'sex')
if (length(unique(md$status)) > 1L) terms <- c(terms, 'status')
formula <- stats::reformulate(terms)
design <- stats::model.matrix(formula, md)
if (qr(design)$rank != ncol(design)) stop('Rank-deficient design matrix')
if (!('carrier1' %in% colnames(design))) stop('Missing carrier1 coefficient')
if (nrow(design) <= ncol(design) + 2L) stop('Insufficient residual degrees of freedom')
if (min(table(md$carrier)) < 2L) stop('Carrier groups too small')

cat('edgeR version:',as.character(utils::packageVersion('edgeR')),'\n')
cat('Design:',deparse(formula),'\n')
cat('Samples:',nrow(md),'carrier1:',sum(md$carrier=='1'),
    'carrier0:',sum(md$carrier=='0'),'input genes:',nrow(count_mat),'\n')

# Standard QL workflow for edgeR 4.x. No estimateDisp() needed under legacy=FALSE.
y <- edgeR::DGEList(counts=count_mat)
keep <- edgeR::filterByExpr(y, design=design, min.count=min_count,
                             min.total.count=min_total, min.prop=min_prop)
cat('filterByExpr retained:',sum(keep),'/',length(keep),'\n')
if (sum(keep) < 10L) stop('Too few genes remain after edgeR filterByExpr')
y <- y[keep,,keep.lib.sizes=FALSE]
y <- edgeR::calcNormFactors(y, method='TMM')
if (any(!is.finite(y$samples$norm.factors))) stop('Nonfinite TMM factors')

# New edgeR v4 quasi-likelihood pipeline; estimates NB dispersion internally.
fit <- edgeR::glmQLFit(y, design=design, robust=use_robust, legacy=FALSE)
qlf <- edgeR::glmQLFTest(fit, coef='carrier1')
tab <- as.data.frame(qlf$table)
if (!all(c('logFC','logCPM','F','PValue') %in% names(tab))) {
  stop('Unexpected edgeR glmQLFTest output columns')
}
if (anyDuplicated(rownames(tab))) stop('Duplicated test gene IDs')

res <- data.frame(
    gene_id=rownames(tab),
    log2FC=as.numeric(tab$logFC),
    logCPM=as.numeric(tab$logCPM),
    stat=as.numeric(tab$F),
    pvalue=as.numeric(tab$PValue),
    padj=stats::p.adjust(as.numeric(tab$PValue),method='BH'),
    stat_type=rep('QL_F',nrow(tab)),
    stringsAsFactors=FALSE
)
# Export library-size audit alongside the result bridge; Python preserves it.
audit <- data.frame(donor=md$donor, carrier=as.character(md$carrier),
                    library_size=as.numeric(y$samples$lib.size),
                    TMM_norm_factor=as.numeric(y$samples$norm.factors),
                    effective_library_size=as.numeric(y$samples$lib.size * y$samples$norm.factors))
utils::write.csv(audit, sub('\\.csv$', '_norm_factors.csv', out_path),row.names=FALSE)
utils::write.csv(res,out_path,row.names=FALSE,na='')
cat('Wrote',nrow(res),'genes; BH-FDR<0.05:',sum(res$padj<0.05,na.rm=TRUE),'\n')
