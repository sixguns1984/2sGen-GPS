#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)

get_arg <- function(name, default=NULL) {
  i <- match(name, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(paste("Missing value for", name))
  args[[i+1]]
}

donor_npy <- get_arg("--expr-donor-npy")
sample_npy <- get_arg("--expr-sample-npy")
donor_ids_file <- get_arg("--donor-ids")
sample_ids_file <- get_arg("--sample-ids")
known_file <- get_arg("--known-covariates")
group <- get_arg("--group")
out_dir <- get_arg("--out-dir")
B <- as.integer(get_arg("--B", "20"))
alpha <- as.numeric(get_arg("--alpha", "0.05"))
seed <- as.integer(get_arg("--seed", "1"))
cores <- as.integer(get_arg("--cores", "8"))

required_args <- c(donor_npy, sample_npy, donor_ids_file, sample_ids_file, group, out_dir)
if (any(vapply(required_args, is.null, logical(1)))) {
  stop("Required: --expr-donor-npy --expr-sample-npy --donor-ids --sample-ids --group --out-dir")
}

required_pkgs <- c("PCAForQTL", "RcppCNPy", "ggplot2", "jsonlite")
missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing) > 0) {
  stop(paste("Missing R packages:", paste(missing, collapse=", "),
             "\nRun: Rscript scripts_amp_pd/install_pca_for_qtl.R"))
}

dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

expr_donor <- RcppCNPy::npyLoad(donor_npy, type="numeric")
expr_sample <- RcppCNPy::npyLoad(sample_npy, type="numeric")
donor_ids <- readLines(donor_ids_file, warn=FALSE)
sample_ids <- readLines(sample_ids_file, warn=FALSE)

if (nrow(expr_donor) != length(donor_ids)) {
  stop(sprintf("Donor row mismatch: expression=%d donor_ids=%d",
               nrow(expr_donor), length(donor_ids)))
}
if (nrow(expr_sample) != length(sample_ids)) {
  stop(sprintf("Sample row mismatch: expression=%d sample_ids=%d",
               nrow(expr_sample), length(sample_ids)))
}
if (ncol(expr_donor) != ncol(expr_sample)) {
  stop("Donor and sample expression matrices have different gene counts.")
}
if (any(!is.finite(expr_donor)) || any(!is.finite(expr_sample))) {
  stop("Expression matrix contains non-finite values.")
}

rownames(expr_donor) <- donor_ids
rownames(expr_sample) <- sample_ids

# Drop features with zero/undefined variance in the donor-level matrix because
# PCAForQTL::runBE() internally calls prcomp(..., scale.=TRUE).
sds <- apply(expr_donor, 2, sd)
keep <- is.finite(sds) & sds > 0
if (sum(keep) < 10) stop("Too few non-constant donor-level genes.")
expr_donor2 <- expr_donor[, keep, drop=FALSE]
expr_sample2 <- expr_sample[, keep, drop=FALSE]

cat(sprintf("[%s] samples=%d donors(BE)=%d genes_input=%d genes_used=%d B=%d alpha=%g cores=%d\n",
            group, nrow(expr_sample2), nrow(expr_donor2), ncol(expr_donor),
            ncol(expr_donor2), B, alpha, cores))

RNGkind("L'Ecuyer-CMRG")
set.seed(seed)

# K selection is participant-balanced.
cat("Running PCAForQTL::runBE() on donor-mean expression ...\n")
be <- PCAForQTL::runBE(expr_donor2, B=B, alpha=alpha, mc.cores=cores)
K_BE <- as.integer(be$numOfPCsChosen)
if (!is.finite(K_BE) || K_BE < 0) stop("Invalid K_BE returned by PCAForQTL::runBE().")

# Fit PCA on the same donor-level matrix used for K selection.
rank_target <- min(max(100L, K_BE), nrow(expr_donor2)-1L, ncol(expr_donor2))
if (rank_target < 1) stop("Donor matrix rank is too small.")
cat("Running donor-level prcomp() with rank =", rank_target, "...\n")
pr <- prcomp(expr_donor2, center=TRUE, scale.=TRUE, rank.=rank_target)
K_elbow <- as.integer(PCAForQTL::runElbow(prcompResult=pr))

if (K_BE > ncol(pr$rotation)) {
  stop(sprintf("K_BE=%d exceeds donor PCA components=%d", K_BE, ncol(pr$rotation)))
}

# Project every longitudinal sample into donor-derived PCA space.
if (K_BE > 0) {
  Xs <- sweep(expr_sample2, 2, pr$center, FUN="-")
  Xs <- sweep(Xs, 2, pr$scale, FUN="/")
  rotation_top <- pr$rotation[, seq_len(K_BE), drop=FALSE]
  pcs_sample <- Xs %*% rotation_top

  # Optional numerical stabilization recommended by PCAForQTL.
  pcs_sample <- scale(pcs_sample)
  colnames(pcs_sample) <- paste0("PC", seq_len(K_BE))
  rownames(pcs_sample) <- sample_ids

  pc_out <- data.frame(
    sample_id=sample_ids,
    pcs_sample,
    check.names=FALSE,
    row.names=NULL
  )
} else {
  pcs_sample <- matrix(numeric(0), nrow=nrow(expr_sample2), ncol=0)
  rownames(pcs_sample) <- sample_ids
  pc_out <- data.frame(sample_id=sample_ids, check.names=FALSE)
}

write.csv(
  pc_out,
  file=file.path(out_dir, "top_BE_PCs.csv"),
  row.names=FALSE, quote=FALSE
)

# Save donor PCA PVE, which is the PCA whose K is actually being selected.
pve <- (pr$sdev^2) / sum(pr$sdev^2)
write.csv(
  data.frame(PC=seq_along(pve), PVE=pve),
  file=file.path(out_dir, "PVE.csv"),
  row.names=FALSE, quote=FALSE
)

# Save BE p-values for direct diagnostic inspection.
write.csv(
  data.frame(PC=seq_along(be$pValues), BE_p_value=be$pValues),
  file=file.path(out_dir, "BE_pvalues.csv"),
  row.names=FALSE, quote=FALSE
)

# Known-covariate R2 uses sample-level projected expression PCs, because these
# are the actual covariates entering the QTL model.
r2_rows <- list()
if (!is.null(known_file) && file.exists(known_file) && K_BE > 0) {
  known <- read.csv(known_file, check.names=FALSE, stringsAsFactors=FALSE)
  if (!"sample_id" %in% colnames(known)) stop("known-covariates file requires sample_id.")
  rownames(known) <- as.character(known$sample_id)
  known <- known[sample_ids, , drop=FALSE]

  cov_names <- setdiff(colnames(known), c("sample_id", "participant_id"))
  for (nm in cov_names) {
    y <- suppressWarnings(as.numeric(known[[nm]]))
    ok <- is.finite(y)
    if (sum(ok) <= (K_BE + 2)) {
      r2 <- NA_real_
    } else {
      x <- pcs_sample[ok, , drop=FALSE]
      fit <- lm(y[ok] ~ x)
      r2 <- summary(fit)$r.squared
    }
    r2_rows[[length(r2_rows)+1L]] <- data.frame(covariate=nm, R2=r2)
  }
}
r2_df <- if (length(r2_rows)) do.call(rbind, r2_rows) else
  data.frame(covariate=character(), R2=numeric())
write.csv(
  r2_df,
  file=file.path(out_dir, "known_covariate_R2.csv"),
  row.names=FALSE, quote=FALSE
)

try({
  PCAForQTL::makeScreePlot(
    prcompResult=pr,
    labels=c("Elbow", "BE"),
    values=c(K_elbow, K_BE),
    titleText=paste0(group, " (participant-balanced)")
  )
  ggplot2::ggsave(
    filename=file.path(out_dir, "scree_BE_elbow.pdf"),
    width=7, height=5, units="in"
  )
}, silent=TRUE)

summary_list <- list(
  group=group,
  n_samples=nrow(expr_sample2),
  n_donors_BE=nrow(expr_donor2),
  n_genes_input=ncol(expr_donor),
  n_genes_used=ncol(expr_donor2),
  B=B,
  alpha=alpha,
  seed=seed,
  cores=cores,
  K_BE=K_BE,
  K_elbow=K_elbow,
  K_over_n_donors=ifelse(nrow(expr_donor2)>0, K_BE/nrow(expr_donor2), NA),
  BE_observation_unit="participant",
  BE_aggregation="mean_across_retained_visits",
  PC_loading_fit="donor_mean_expression",
  sample_PC_scoring="projection_onto_donor_PCA_loadings",
  PCAForQTL_version=as.character(utils::packageVersion("PCAForQTL")),
  R_version=R.version.string
)
jsonlite::write_json(
  summary_list,
  path=file.path(out_dir, "BE_summary.json"),
  pretty=TRUE, auto_unbox=TRUE
)

capture.output(sessionInfo(), file=file.path(out_dir, "sessionInfo.txt"))

cat(sprintf("[%s] donors=%d samples=%d K_BE=%d K_elbow=%d\n",
            group, nrow(expr_donor2), nrow(expr_sample2), K_BE, K_elbow))
cat("Outputs:", out_dir, "\n")
