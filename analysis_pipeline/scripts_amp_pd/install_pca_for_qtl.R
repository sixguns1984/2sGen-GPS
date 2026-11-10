#!/usr/bin/env Rscript
repos <- "https://cloud.r-project.org"

cran_pkgs <- c("remotes", "RcppCNPy", "ggplot2", "jsonlite")
for (p in cran_pkgs) {
  if (!requireNamespace(p, quietly=TRUE)) {
    install.packages(p, repos=repos)
  }
}

if (!requireNamespace("PCAForQTL", quietly=TRUE)) {
  remotes::install_github("heatherjzhou/PCAForQTL")
}

cat("PCAForQTL:", as.character(packageVersion("PCAForQTL")), "\n")
cat("RcppCNPy:", as.character(packageVersion("RcppCNPy")), "\n")
cat("ggplot2:", as.character(packageVersion("ggplot2")), "\n")
cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
