# Cell-Cell Communication Analysis Pipeline

Age-related changes in ligand-receptor signaling across 11 brain regions in non-human primates (NHP), analyzed at single-cell resolution using LIANA + CellPhoneDB v5.

## Dataset
- 55 rhesus macaques (age range 2.5–21 years)
- 11 brain regions: ACC, CN, dlPFC, EC, HIP, IPP, lCb, M1, MB, mdTN, NAc
- 12 cell types: Astrocyte, GABA, Glutamatergic, Microglia, Oligo, OPC, Vascular, Basket, Cerebellar, Ependymal, Midbrain, MSN
- ~2,716 ligand-receptor pairs from CellPhoneDB v5

## Pipeline Overview
```
Per-animal snRNA-seq (h5ad)
        ↓
Step 1: Per-animal LIANA/CellPhoneDB v5 CCC
        run_per_animal_louvain_v2.py
        → results/per_animal_louvain_v2/
        ↓
Step 2: Build LR means matrix per region
        build_lr_matrix.py
        → results/lr_matrices/
        ↓
Step 3: Whole-region OLS regression (lr_means ~ age + sex)
        run_age_sex_regression.py
        → results/within_region_analysis/regression_{Region}/
        ↓
Step 4a: mashr (NaN-filter approach)        Step 4b: mashr (top-N per region)
         run_mashr_ccc.R                              run_mashr_ccc_topn.R
         → results/mashr/{CT}_sender/receiver_        → results/mashr/{CT}_sender/receiver_
           louvain_nanfilt{X}/                          louvain_top8perregion/
        ↓
Step 5: Visualization & summary tables
        plot_*.py, build_mashr_*.py
        → results/mashr/plots/
          results/mashr/plots_top8perregion/
          results/regression_plots/
```

## Key Scripts

### CCC Pipeline
| Script | Description |
|--------|-------------|
| `run_per_animal_louvain_v2.py` | Per-animal LIANA CCC using CellPhoneDB v5. Runs on HPC via SLURM. Validates 2,716 LR pairs at startup. Parameters: `expr_prop=0.05`, FDR on `cellphone_pvals` |
| `build_lr_matrix.py` | Builds lr_means matrix from per-animal CSV files. Args: `--region`, `--threshold`, `--min_age`, `--input_dir`, `--output_name` |
| `run_age_sex_regression.py` | OLS regression (`lr_means ~ age + sex`) per interaction. Args: `--region`, `--min_animals` (default 10), `--min_age`, `--exclude_animals`, `--matrix_file`, `--output_suffix` |

### mashr
| Script | Description |
|--------|-------------|
| `run_mashr_ccc.R` | mashr on age coefficients from regression. Conditions = Louvain combos filtered by NaN rate. Args: `--source_ct` or `--target_ct`, `--louvain`, `--nan_filter` |
| `run_mashr_ccc_topn.R` | mashr with balanced regional coverage. Selects top N densest Louvain combos per region. Args: `--source_ct` or `--target_ct`, `--top_n` (default 8) |
| `run_mashr_louvain.R` | mashr for a specific source→target Louvain pair across regions. Args: source_louvain, target_louvain |
| `compare_mashr_covariance.R` | Compares covariance structures between nanfilt and top8 runs |

### Annotations
| Script | Description |
|--------|-------------|
| `build_cpdb_annotations.py` | Builds `cpdb_lr_annotations.csv` — 2,716 LR pairs with 19 broad functional categories |
| `cpdb_lr_annotations.csv` | LR pair annotations: `lr_pair`, `classification`, `broad_category` |

### Summary Tables
| Script | Description |
|--------|-------------|
| `build_mashr_master_table_top8.py` | Master table: lr_pair × sender→receiver × n_sig conditions, tier, mean_beta |
| `build_mashr_region_table_top8.py` | Region-level table: sender_ct × receiver_ct × region aggregated from Louvain mashr |
| `build_mashr_comparison_tables.py` | Comparison tables: nanfilt vs top8 overlap, regional coverage, sharing distribution |

### Visualization
| Script | Description |
|--------|-------------|
| `plot_regression_region_celltype_heatmap.py` | Heatmap: mean age_coef per cell type pair × region (all LR pairs, asterisks for sig) |
| `plot_regression_per_region_dotplot.py` | Per-region dot plot from regression outputs |
| `plot_mashr_per_region_dotplot_top8.py` | Per-region dot plot from mashr top8 results |
| `plot_mashr_bubble_matrix_top8.py` | Bubble matrix: sender × receiver, size=n sig LR pairs, color=direction, opacity=specificity |
| `plot_mashr_beta_heatmap_bytier_top8.py` | Heatmap of LR pairs × conditions by sharing tier (global/broad/intermediate/specific) |
| `plot_mashr_beta_heatmap_all_top8.py` | Heatmap of all LR pairs × conditions (two versions: 95th pct scale and capped) |
| `plot_mashr_category_enrichment_top8.py` | Functional category enrichment per sender, per sharing tier (Fisher's exact, FDR) |
| `plot_mashr_sharing_table_top8.py` | Summary table: sharing distribution per sender cell type |
| `plot_mashr_scatter_sanity.py` | Sanity check scatter plots: lr_means vs age per Louvain combo condition |
| `plot_mashr_condition_specific_top8.py` | Table of condition-specific hits (sig in ≤5 conditions) |
| `plot_mashr_regional_dotplot_top8.py` | Regional dot plot: LR pairs × regions, size=n sig Louvain combos |
| `plot_mashr_summary_heatmap_top8.py` | Cross-sender summary heatmap |

## Output Structure
```
results/
├── per_animal_louvain_v2/          # Per-animal LIANA outputs
├── per_animal_louvain_dlpfc_attempt2/  # dlPFC rerun
├── lr_matrices/                    # LR means matrices per region
├── within_region_analysis/         # Regression outputs
│   └── regression_{Region}/
│       └── whole_{region}_age_sex_regression.csv
├── mashr/                          # mashr outputs
│   ├── {CT}_sender_louvain_nanfilt{X}/
│   ├── {CT}_receiver_louvain_nanfilt{X}/
│   ├── {CT}_sender_louvain_top8perregion/
│   ├── {CT}_receiver_louvain_top8perregion/
│   ├── mashr_master_table_top8.csv
│   ├── mashr_region_table_top8.csv
│   ├── plots/                      # nanfilt visualizations
│   ├── plots_top8perregion/        # top8 visualizations
│   └── plots_comparison/           # comparison tables
├── regression_plots/               # Regression-based visualizations
│   └── per_region/
└── mashr_test/                     # Early test runs
```

## Regression Output Format

`whole_{region}_age_sex_regression.csv` columns:
- `interaction`: `source_louvain|target_louvain|ligand_complex|receptor_complex`
- `n_animals`: number of animals with data for this interaction
- `mean_lr_means`: mean lr_means across animals
- `age_coef`, `age_stderr`, `age_pval`, `age_qval`: age effect (OLS, BH-FDR)
- `sex_coef`, `sex_stderr`, `sex_pval`, `sex_qval`: sex effect
- `r_squared`: model R²

## mashr Output Format

`mashr_{label}_results.csv` columns:
- `lr_pair`: `ligand|receptor`
- `condition`: `source_louvain|target_louvain|region`
- `beta_posterior`: mashr posterior mean effect size
- `sd_posterior`: posterior SD
- `lfsr`: local false sign rate (analogous to FDR)
- `beta_original`: original OLS age_coef (NA if condition was missing)
- `se_original`: original OLS age_stderr (NA if condition was missing)

## Environment
```bash
conda activate cellchat_env    # CCC pipeline (Python)
conda activate mashr_env       # mashr (R)
```

## Key Parameters

- **CCC**: `expr_prop=0.05`, FDR correction on `cellphone_pvals`, min 1 animal per interaction
- **Regression**: `min_animals=10`, `min_age=1.0` (excludes infants)
- **mashr NaN filter**: keep conditions where ≤55% of LR pairs are missing (varies by cell type)
- **mashr top-N**: top 8 densest Louvain combos per region (guarantees all 11 regions represented)
- **Significance**: `age_qval < 0.05` (regression), `lfsr < 0.05` (mashr)
