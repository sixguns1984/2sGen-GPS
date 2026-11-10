# Current V8.9 analysis flow

```text
Raw AMP-PD resources
│
├─ Stage-I manuscript-exact resources
│  ├─ exact PPMI/PDBP longitudinal samples
│  ├─ exact CIBERSORTx LM22 cell-composition covariates
│  └─ manuscript benchmark/replication methods
│
├─ Current-V8.9 longitudinal resources
│  ├─ PP_FULL / PD_FULL expression/genotype/covariate resources
│  ├─ expression PCs and cis map
│  └─ CTAR-MPTGA temporal cis-QTL discovery + PDBP replication
│
├─ Stage-I sensitivity/validation
│  ├─ simulation benchmark
│  └─ residual-vQTL sensitivity for the dynamic subclass
│
└─ Stage II: current IV-informed temporal phenotype pipeline
   │
   ├─ shared IV preparation
   │  ├─ PPMI temporal BH-FDR < 0.05
   │  ├─ same SNP-gene PDBP temporal P < 0.05
   │  ├─ trajectory_r > 0
   │  ├─ within-gene PPMI LD pruning (r2 < 0.10)
   │  └─ first-stage strength diagnostics (annotation only)
   │
   ├─ phenotype-specific primary analysis
   │  ├─ phenotype/genotype overlap + SNP extraction
   │  ├─ cohort-specific genotype PCs
   │  ├─ frozen single-IV temporal component at phenotype visits
   │  ├─ cumulative lag 1–4 FAST analysis
   │  ├─ one PPMI global IV×lag BH-FDR
   │  └─ exact same IV-gene replication in PD and SU (nominal P < 0.05)
   │
   └─ post-selection robustness (never changes membership)
      ├─ selected-lag full cluster details + AR-type test
      ├─ all Stage-I independent IVs at one common gene lag
      ├─ fixed-effect pooled estimate, Q/I2, LOO
      ├─ cross-gene LD audit
      └─ final causal IV-gene master table
```

There is no separate production Stage III. MR-Egger/scalarized MR and coloc are not part of the standard pipeline.

See `README.md` for exact commands and output paths.
