# Run interactively in RStudio, after regenerating the coef_sum POWER80 table.
library(dplyr)
library(ggplot2)
library(stringr)
library(enrichR)

file <- "/public2/data/luojunfeng/2sGenGPS_revised_chatgpt/single_iv_causal/cross_phenotype_summary_lag1/Supplement_PD_IV_gene_publication_3phenotypes_POWER80.csv"
# On macOS/RStudio, change file to your copied CSV path, or use: file <- file.choose()
df <- read.csv(file, check.names = FALSE) %>%
  filter(phenotype %in% c("updrs1", "updrs2", "updrs3"), !is.na(PPMI_coef_sum), PPMI_coef_sum != 0)

# Direction follows the aggregate lag coefficient, not lag-1 or expression up/downregulation.
pos <- unique(df$gene[df$PPMI_coef_sum > 0])
neg <- unique(df$gene[df$PPMI_coef_sum < 0])
cat("Positive genes:", length(pos), " Negative genes:", length(neg), "\n")
cat("Genes shared across directions:", paste(intersect(pos,neg),collapse=", "), "\n")

db <- grep("^GO_Biological_Process", listEnrichrDbs()$libraryName, value=TRUE)
if (!length(db)) stop("No GO_Biological_Process database available from Enrichr")
db <- sort(db, decreasing=TRUE)[1]
cat("Using Enrichr database:", db, "\n")

x1 <- enrichr(pos, db)[[1]] %>% mutate(Direction="Positive aggregate association")
x2 <- enrichr(neg, db)[[1]] %>% mutate(Direction="Negative aggregate association")
go <- bind_rows(x1, x2) %>%
  filter(!is.na(P.value), P.value >= 0, !is.na(Genes), Genes != "") %>%
  arrange(Direction, P.value)

# Top 5 GO BP terms in each direction; remove near-duplicates using Jaccard overlap
# of the genes actually overlapping each term. No adjusted-P cutoff is imposed.
keep <- list()
for (dir in c("Positive aggregate association", "Negative aggregate association")) {
  x <- go %>% filter(Direction == dir) %>% arrange(P.value)
  chosen <- integer(0)
  for (i in seq_len(nrow(x))) {
    a <- unique(trimws(strsplit(x$Genes[i], ";", fixed=TRUE)[[1]]))
    overlaps <- vapply(chosen, function(j) {
      b <- unique(trimws(strsplit(x$Genes[j], ";", fixed=TRUE)[[1]]))
      length(intersect(a,b)) / max(1, length(union(a,b)))
    }, numeric(1))
    if (!length(overlaps) || all(overlaps < 0.5)) chosen <- c(chosen, i)
    if (length(chosen) == 5) break
  }
  keep[[dir]] <- x[chosen, ]
}

plot_df <- bind_rows(keep) %>%
  mutate(
    Direction = factor(Direction, levels=c("Positive aggregate association", "Negative aggregate association")),
    neglogP = -log10(pmax(P.value, 1e-300)),
    term_label = str_wrap(gsub(" \\(GO:[0-9]+\\)$", "", Term), 42),
    gene_label = vapply(strsplit(Genes, ";", fixed=TRUE),
       function(z) paste(head(trimws(z), 3), collapse=", "), character(1)),
    term_id = paste(as.character(Direction), term_label, sep="___")
  ) %>% arrange(Direction, desc(neglogP))

if (!nrow(plot_df)) stop("Enrichr returned no annotated terms")
plot_df$term_id <- factor(plot_df$term_id, levels=rev(unique(plot_df$term_id)))

p <- ggplot(plot_df, aes(x=neglogP, y=term_id, fill=Direction)) +
  geom_col(width=0.70, show.legend=FALSE) +
  geom_text(aes(x=neglogP*0.035, label=gene_label),
            size=2.65, color="white", hjust=0, lineheight=0.95) +
  facet_grid(Direction ~ ., scales="free_y", space="free_y", switch="y") +
  scale_fill_manual(values=c("Positive aggregate association"="#CB6B50",
                             "Negative aggregate association"="#39799C")) +
  scale_y_discrete(labels=function(x) sub("^.*___", "", x)) +
  scale_x_continuous(expand=expansion(mult=c(0,0.07))) +
  labs(x=expression(-log[10]("Nominal P (Enrichr)")), y=NULL) +
  theme_classic(base_size=10) +
  theme(
    legend.position="none", strip.background=element_blank(),
    strip.text.y.left=element_text(size=9.5, face="bold", angle=90),
    strip.placement="outside", axis.text.y=element_text(size=8.5, color="#242B30"),
    axis.text.x=element_text(size=8.5, color="#242B30"),
    axis.line.y=element_blank(), axis.ticks.y=element_blank(),
    panel.spacing.y=grid::unit(0.5,"cm"),
    plot.margin=margin(7,14,7,7)
  )
print(p)

ggsave("PD_UPDRS_coefsum_GO_BP.pdf", p, width=8.1, height=5.0, useDingbats=FALSE)
ggsave("PD_UPDRS_coefsum_GO_BP.png", p, width=8.1, height=5.0, dpi=600)
write.csv(go, "PD_UPDRS_coefsum_GO_BP_all.csv", row.names=FALSE)
write.csv(plot_df, "PD_UPDRS_coefsum_GO_BP_top5_nonredundant.csv", row.names=FALSE)
writeLines(pos, "PD_coefsum_positive_genes.txt")
writeLines(neg, "PD_coefsum_negative_genes.txt")
