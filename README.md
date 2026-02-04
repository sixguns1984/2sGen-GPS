# 2sGen-GPS: Two-Stage Genetic Granger Temporal Causality Study

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-active-success)]()
[![DOI](https://img.shields.io/badge/DOI-TBD-orange)]()

## 🧬 Overview
2sGen-GPS (two-stage genetic Granger temporal causality study) is a novel multi-omics/multi-modal data integrative analysis framework designed for longitudinal cohort studies. The method enables:

1. Genome-wide temporal eQTL (teQTL) mapping - Identifying genetic variants that regulate gene expression over time

2. Dynamic eQTL detection - Uncovering eQTLs exhibiting significant temporal variation in stationary longitudinal gene expressions

3. Temporal causality inference - Establishing causal relationships between longitudinal gene expression and phenotypes using teQTLs as instrumental variables

## 📊 Methodological Framework

### Stage 1: Temporal eQTL Mapping
Input: Longitudinal multi-omics data (genotype + time-series gene expression)

Approach: Statistical modeling of temporal genetic regulation

Output: Significant teQTLs with time-varying effects

### Stage 2: Granger Causality Analysis
Input: teQTLs + longitudinal phenotypes

Approach: Two-stage least squares (2SLS) with Granger causality testing

Output: Causal relationships between gene expression dynamics and phenotypic trajectories

![Schematic of the 2sGen-GPS approaches](https://github.com/sixguns1984/2sGen-GPS/blob/main/flowchart.png)

## 📁 Repository Structure

```bash
2sGen-GPS/
├── mptga_main.py              # Main script for temporal/dynamic eQTL mapping
├── teQTL_powertest_main.py    # Power analysis for teQTL detection
├── twosGen_GPS_main.py        # Two-stage Granger causality analysis
├── poly_regression_main.py    # Linear/cubic polynomial regression for teQTL analysis
├── AR1_main.py               # First-order autoregression for teQTL analysis
├── LICENSE                    # License file
└── example/                   # Example datasets
```

Corresponding with: Ganqiang Liu (liugq3@mail.sysu.edu.cn); Junfeng Luo (luojf26@mail2.sysu.edu.cn)
