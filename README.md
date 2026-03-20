# Cell-Cell Communication Analysis Pipeline

Age-related changes in ligand-receptor signaling across 11 brain regions in non-human primates (NHP), analyzed at single-cell resolution using LIANA + CellPhoneDB v5.

## Dataset
- 55 rhesus macaques (age range 2.5–21 years)
- 11 brain regions: ACC, CN, dlPFC, EC, HIP, IPP, lCb, M1, MB, mdTN, NAc
- 12 cell types: Astrocyte, GABAergic, Glutamatergic, Microglia, Oligodendrocyte, OPC, Vascular, Basket, Cerebellar, Ependymal, Midbrain, MSN
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
Step 4: Hypergeometric enrichment by cell type × role × direction
        hypergeometric_enrichment_all_regions.py
        → results/within_region_analysis/hypergeometric_all_regions/
        ↓
Step 5: Visualization
        plot_*.py
        → results/within_region_analysis/hypergeometric_all_regions/
          results/regression_plots/
```

## Key Scripts

### CCC Pipeline
| Script | Description |
|--------|-------------|
| `run_per_animal_louvain_v2.py` | Per-animal LIANA CCC using CellPhoneDB v5. Runs on HPC via SLURM. Validates 2,716 LR pairs at startup. Parameters: `expr_prop=0.05`, FDR on `cellphone_pvals` |
| `build_lr_matrix.py` | Builds lr_means matrix from per-animal CSV files. Args: `--region`, `--threshold`, `--min_age`, `--input_dir`, `--output_name` |
| `run_age_sex_regression.py` | OLS regression (`lr_means ~ age + sex`) per interaction. Args: `--region`, `--min_animals` (default 10), `--min_age`, `--exclude_animals`, `--matrix_file`, `--output_suffix` |

### Enrichment Analysis
| Script | Description |
|--------|-------------|
| `hypergeometric_enrichment_all_regions.py` | Hypergeometric enrichment and depletion tests for age-associated LR pairs by cell type × role (sender/receiver) × direction (strengthening/weakening) across all 11 regions. Single global BH-FDR correction across all 664 tests. |

### Annotations
| Script | Description |
|--------|-------------|
| `build_cpdb_annotations.py` | Builds `cpdb_lr_annotations.csv` — 2,716 LR pairs with 19 broad functional categories |
| `cpdb_lr_annotations.csv` | LR pair annotations: `lr_pair`, `classification`, `broad_category` |

### Visualization
| Script | Description |
|--------|-------------|
| `plot_regression_region_celltype_heatmap.py` | Heatmap: mean age_coef per cell type pair × region (all LR pairs, asterisks for sig) |
| `plot_regression_per_region_dotplot.py` | Per-region dot plot from regression outputs |
| `plot_hypergeometric_heatmaps.py` | Heatmaps of fold enrichment by cell type × region, for each direction × role combination. Args: `--log` (log2/log10), `--clip` (clip at FE=1), `--sig_only` (only show significant cells) |

## Output Structure
```
results/
├── per_animal_louvain_v2/          # Per-animal LIANA outputs
├── per_animal_louvain_dlpfc_attempt2/  # dlPFC rerun
├── lr_matrices/                    # LR means matrices per region
├── within_region_analysis/         # Regression + enrichment outputs
│   ├── regression_{Region}/
│   │   └── whole_{region}_age_sex_regression.csv
│   └── hypergeometric_all_regions/
│       └── hypergeometric_enrichment_all_regions.csv
└── regression_plots/               # Regression-based visualizations
    └── per_region/
```

## Regression Output Format

`whole_{region}_age_sex_regression.csv` columns:
- `interaction`: `source_louvain|target_louvain|ligand_complex|receptor_complex`
- `n_animals`: number of animals with data for this interaction
- `mean_lr_means`: mean lr_means across animals
- `age_coef`, `age_stderr`, `age_pval`, `age_qval`: age effect (OLS, BH-FDR)
- `sex_coef`, `sex_stderr`, `sex_pval`, `sex_qval`: sex effect
- `r_squared`: model R²

## Hypergeometric Enrichment Output Format

`hypergeometric_enrichment_all_regions.csv` columns:
- `region`: brain region
- `cell_type`: broad cell type (Louvain clusters collapsed)
- `role`: `sender` or `receiver`
- `n_tested`: total interactions tested for this cell type × role
- `k_pos`, `expected_pos`, `fold_enrichment_pos`: observed, expected, and fold enrichment for strengthening interactions
- `k_neg`, `expected_neg`, `fold_enrichment_neg`: observed, expected, and fold enrichment for weakening interactions
- `p_pos_enrich`, `p_pos_deplete`: raw p-values for strengthening enrichment/depletion
- `p_neg_enrich`, `p_neg_deplete`: raw p-values for weakening enrichment/depletion
- `q_pos_enrich`, `q_pos_deplete`, `q_neg_enrich`, `q_neg_deplete`: BH-FDR corrected q-values

## Environment
```bash
conda activate cellchat_env    # CCC pipeline (Python)
```

## Key Parameters

- **CCC**: `expr_prop=0.05`, FDR correction on `cellphone_pvals`, min 1 animal per interaction
- **Regression**: `min_animals=10`, `min_age=1.0` (excludes infants)
- **Enrichment**: `age_qval < 0.05` (regression significance threshold), hypergeometric test with global BH-FDR correction across 664 tests
