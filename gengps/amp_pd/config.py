from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AmpPdPaths:
    root: str = "/public2/data/luojunfeng/2sGenGPS_revised_chatgpt"
    gene_annotation_csv: str = "/public2/data/luojunfeng/2sGenGPS/tQTL/dataset/gencode.v29.primary_assembly.annotation.csv"
    rin_csv: str = "/public/labdata/public_data/AMP_PD_releases2021_v2-5release_0510_RNA/releases_2021_v2-5release_0510_rnaseq_sequencing_metrics_rna_quality_metrics.csv"
    matrix_genes_tsv: str = "/public/labdata/public_data/AMP_PD_releases2021_v2-5release_0510_RNA/matrix.genes.tsv"
    demographics_csv: str = "/public/labdata/public_data/AMP_PD_clinical/AMP_PD_releases2022_V3release_clinical/releases_2022_v3release_1115_clinical_Demographics.csv"
    medical_history_csv: str = "/public/labdata/public_data/AMP_PD_clinical/AMP_PD_releases2022_V3release_clinical/releases_2022_v3release_1115_clinical_PD_Medical_History.csv"
    wgs_bfile: str = "/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged"
    plink19: str = "plink1.9"
    twstats: str = "twstats"
    twtable: str = "twtable"

    @property
    def root_path(self): return Path(self.root)
    @property
    def data_dir(self): return self.root_path / "data"
    @property
    def qtl_dir(self): return self.root_path / "qtl"
    @property
    def logs_dir(self): return self.root_path / "logs"

COHORT_CODE_TO_NAME = {
    "PP": "PPMI",
    "PD": "PDBP",
    "BF": "BioFIND",
    "REP": "PDBP_BioFIND",
}

VISIT_TO_YEARS = {
    "SVM0_5T1": 0.0,
    "BLM0T1": 0.0,
    "SVM6T1": 0.5,
    "SVM12T1": 1.0,
    "SVM18T1": 1.5,
    "SVM24T1": 2.0,
    "SVM36T1": 3.0,
}

LEGACY_TIMES_BY_COHORT = {
    "PP": [0.0, 0.5, 1.0, 2.0, 3.0],
    "PD": [0.0, 0.5, 1.0, 1.5, 2.0],
}

# Genotype QC/PCA is no longer pooled across all AMP-PD RNA participants.
# Genotype QC: PPMI discovery and a shared REP (PDBP + BioFIND) genotype resource.
# Primary follow-up-time tQTL replication itself is PPMI -> PDBP only.
GENOTYPE_QC_GROUPS = {
    "PP": ["PP"],
    "REP": ["PD", "BF"],
}

# QTL expression universe and CIBERSORTx universe are intentionally separated.
# QTL uses the stricter TPM>=1 universe; CIBERSORTx keeps broad raw TPM genes.
DEFAULT_QTL_MEAN_TPM = 1.0
DEFAULT_DECONV_MEAN_TPM = 0.01
DEFAULT_TMP_DIR = "/public/labdata/luojunfeng/tmp"
DEFAULT_MAX_PCS = 20
DEFAULT_MANUAL_GPCS = {"PP": 8, "REP": 8}
# Expression-PC count is NEVER hard-coded in v8. It is selected separately
# for each analysis sample set by PCAForQTL::runBE().
EXPRESSION_PC_BE_GROUPS = {
    "PP_FULL": {"cohorts": ["PP"], "legacy": False},
    "PP_LEGACY": {"cohorts": ["PP"], "legacy": True},
    "PD_LEGACY": {"cohorts": ["PD"], "legacy": True},
    # All available PDBP RNA visits only. This is the preferred expression-PC
    # nuisance set for revised PPMI->PDBP follow-up analyses; it avoids using
    # BioFIND expression to define PDBP nuisance axes.
    "PD_FULL": {"cohorts": ["PD"], "legacy": False},
    "REP_FULL": {"cohorts": ["PD", "BF"], "legacy": False},
}
DEFAULT_BE_B = 20
DEFAULT_BE_ALPHA = 0.05
DEFAULT_BE_SEED = 1
DEFAULT_BE_CORES = 8

SUBMITTED_BASE_COVARIATES = [
    "age_at_baseline",
    "sex_binary",
    "diagnosis_binary",
    "RIN",
]

REVISED_BASE_COVARIATES = [
    "sex_binary",
    "diagnosis_binary",
    "RIN",
]

CELL_COMPOSITION_COVARIATES = [
    "cibx_clr_B_z",
    "cibx_clr_CD4T_z",
    "cibx_clr_CD8T_z",
    "cibx_clr_NK_z",
    "cibx_clr_Monocytes_z",
    "cibx_clr_Neutrophils_z",
]

DISCOVERY_COHORT = "PP"
FOLLOWUP_REPLICATION_COHORT = "PD"
AGE_REPLICATION_COHORTS = ["REP"]

FOLLOWUP_METHODS = [
    "submitted_followup_mptga",
    "lin2018_mptga",
    "lin2018_regression",
    "lin2018_ar",
    "union",
    "fisher",
    "manova",
]

# Revised raw-data longitudinal methods. These are kept separate from the
# manuscript-exact method list so legacy Stage-I reruns remain reproducible.
REVISED_FOLLOWUP_METHODS = [
    "submitted_followup_ctar_mptga",
    "lin2018_regression_cov",
]

ALL_FOLLOWUP_METHODS = FOLLOWUP_METHODS + REVISED_FOLLOWUP_METHODS

AGE_METHODS = [
    "proposed_age_cubic",
    "proposed_age_spline",
    "age_cubic_iid_cov",
]

# All production longitudinal Stage-I methods, including the two revised
# raw-derived methods.  Historical manuscript-exact and revised methods retain
# separate wrappers/data modes but are both part of the current benchmark.
ALL_DISCOVERY_METHODS = list(ALL_FOLLOWUP_METHODS)

# Age-QTL implementations are retained only as optional/exploratory modules.
# The primary Stage-I workflow is strictly longitudinal follow-up-time tQTL mapping.
OPTIONAL_AGE_METHODS = list(AGE_METHODS)

# Method implementation fingerprints used to protect resumability when an
# algorithm changes but the result directory name remains the same. Existing
# chunks/finals are reused only when the fingerprint matches.
METHOD_IMPLEMENTATION_TAGS = {
    "lin2018_mptga": "lin2018_complete_grid_noimpute_matched_v2",
}

# Stage-I QTL code/data fingerprints.  The manuscript mode is deliberately
# separated from the current V8.9 preprocessing used by Stage-II/III.
STAGE1_MANUSCRIPT_IMPLEMENTATION_TAGS = {
    "submitted_followup_mptga": "stage1_manuscript_exact_v5__submitted_mptga_ar1suffstats_binary_exactcibx_rawpdbp",
    "lin2018_mptga": "stage1_manuscript_exact_v5__lin2018_mptga_ar1suffstats_binary_exactcibx_rawpdbp",
    "lin2018_regression": "stage1_manuscript_exact_v4__lin2018_regression_binary_exactcibx_rawpdbp",
    "lin2018_ar": "stage1_manuscript_exact_v4__lin2018_ar_binary_exactcibx_rawpdbp",
    "union": "stage1_manuscript_exact_v4__union_binary_exactcibx_rawpdbp",
    "fisher": "stage1_manuscript_exact_v4__fisher_binary_exactcibx_rawpdbp",
    "manova": "stage1_manuscript_exact_v4__manova_binary_exactcibx_rawpdbp",
}

CURRENT_V89_IMPLEMENTATION_TAGS = {
    "submitted_followup_mptga": "current_v89__submitted_followup_mptga",
    "submitted_followup_ctar_mptga": "current_v89__submitted_followup_ctar_mptga_v1__rawderived_allvisits_cov_cell",
    "lin2018_regression_cov": "current_v89__lin2018_regression_cov_v1__rawderived_allvisits_cov_cell",
    "lin2018_mptga": "current_v89__lin2018_complete_grid_noimpute_matched_v2",
    "lin2018_regression": "current_v89__lin2018_regression",
    "lin2018_ar": "current_v89__lin2018_ar",
    "union": "current_v89__union",
    "fisher": "current_v89__fisher",
    "manova": "current_v89__manova",
}


def implementation_tag(method: str, data_mode: str = "stage1_manuscript", stage1_cell_mode: str = "primary"):
    """Return the implementation fingerprint for resumability/provenance.

    ``stage1_cell_mode='primary'`` is the production Stage-I model with the six
    CIBERSORTx broad-cell CLR covariates. ``'nocell_audit'`` is an internal
    calibration run that uses the otherwise identical manuscript model but
    omits those six covariates.  The audit gets a distinct fingerprint so its
    chunks can never be confused with the primary analysis.
    """
    if data_mode == "stage1_manuscript":
        tag = STAGE1_MANUSCRIPT_IMPLEMENTATION_TAGS.get(method)
        if stage1_cell_mode == "primary":
            return tag
        if stage1_cell_mode == "nocell_audit":
            if method != "submitted_followup_mptga":
                raise ValueError("stage1_cell_mode='nocell_audit' is defined only for submitted_followup_mptga")
            return "stage1_manuscript_exact_v6__submitted_mptga_ar1suffstats_binary_NOCELL_AUDIT_rawpdbp"
        raise ValueError(f"Unknown stage1_cell_mode={stage1_cell_mode!r}")
    if data_mode == "current_v89":
        return CURRENT_V89_IMPLEMENTATION_TAGS.get(method, METHOD_IMPLEMENTATION_TAGS.get(method))
    raise ValueError(f"Unknown data_mode={data_mode!r}")
