# Cell-Cell Communication Analysis Pipeline

Age-related changes in ligand-receptor signaling across 11 brain regions in non-human primates (NHP), analyzed at single-cell resolution using LIANA + CellPhoneDB v5.

## Dataset
- 53 rhesus macaques (age range 2.5–21 years)
- 11 brain regions: ACC, CN, dlPFC, EC, HIP, IPP, lCb, M1, MB, mdTN, NAc
- 12 cell types: Astrocyte, GABAergic, Glutamatergic, Microglia, Oligodendrocyte, OPC, Vascular, Basket, Cerebellar, Ependymal, Midbrain, MSN
- 2,909 ligand-receptor pairs from CellPhoneDB v5 (corrected resource with complexes intact)

## Pipeline Overview
```
Per-animal snRNA-seq (h5ad)
        ↓
Step 1: Per-animal LIANA/CellPhoneDB v5 CCC
        run_per_animal_louvain_ccc.py
        → results/per_animal_louvain_corrected/
        ↓
Step 2: Build LR means matrix per region
        build_lr_matrix.py
        → results/lr_matrices/
        ↓
Step 3: Whole-region OLS regression (lr_means ~ age + sex)
        run_age_sex_regression.py
        → results/within_region_analysis_corrected/regression_results/regression_{Region}/
        ↓
Step 4a: Hypergeometric enrichment by cell type × role × direction
         hypergeometric_enrichment_all_regions.py
         → results/within_region_analysis_corrected/hypergeometric_all_regions/
        ↓
Step 4b: Hypergeometric enrichment by functional LR category × direction
         hypergeometric_category_enrichment.py
         → results/within_region_analysis_corrected/hypergeometric_category/
        ↓
Step 4c: Build per-category LR tables
         build_category_lr_tables.py
         → results/within_region_analysis_corrected/hypergeometric_category/category_tables/
        ↓
Step 5: Visualization
        plotting_scripts/plot_*.py
```

## Key Scripts

### CCC Pipeline
| Script | Description |
|--------|-------------|
| `run_per_animal_louvain_ccc.py` | Per-animal LIANA CCC using CellPhoneDB v5. Runs on HPC via SLURM. Validates 2,909 LR pairs at startup. Parameters: `expr_prop=0.05`, FDR on `cellphone_pvals` |
| `build_lr_matrix.py` | Builds lr_means matrix from per-animal CSV files. Args: `--region`, `--threshold`, `--min_age`, `--input_dir`, `--output_name` |
| `run_age_sex_regression.py` | OLS regression (`lr_means ~ age + sex`) per interaction. Args: `--region`, `--min_animals` (default 10), `--min_age`, `--exclude_animals`, `--matrix_file`, `--output_suffix` |

### Annotations
| File/Script | Description |
|--------|-------------|
| `build_cpdb_annotations.py` | Builds `cpdb_lr_annotations.csv` from CellPhoneDB source files. Three-source lookup: uniprot → special identifiers → manual. Produces 2,909 LR pairs across 21 broad functional categories. |
| `CellPhoneDB-data/interaction_input_CellPhoneDB.csv` | Raw CellPhoneDB interaction table (2,911 rows) |
| `CellPhoneDB-data/gene_input.csv` | Uniprot → gene symbol mapping |
| `CellPhoneDB-data/cellphonedb_interactions_liana_format.csv` | LIANA-format LR pairs (ligand, receptor columns) |
| `cpdb_lr_annotations.csv` | Final LR pair annotations: `ligand`, `receptor`, `lr_pair`, `classification`, `broad_category`, `source` |
| `lr_functional_annotations.py` | Auto-generated Python file with `LR_FUNCTIONS`, `FUNC_COLORS`, `FUNC_ORDER` dicts for plotting |

### Enrichment Analysis
| Script | Description |
|--------|-------------|
| `hypergeometric_enrichment_all_regions.py` | Hypergeometric enrichment and depletion tests for age-associated LR pairs by cell type × role (sender/receiver) × direction (strengthening/weakening) across all 11 regions. Single global BH-FDR correction. |
| `hypergeometric_category_enrichment.py` | Hypergeometric enrichment and depletion tests for age-associated LR pairs by functional category × direction across all 11 regions. Single global BH-FDR correction across all tests. Uses `cpdb_lr_annotations.csv`. |
| `build_category_lr_tables.py` | Builds per-category CSV tables of all age-significant LR pairs for use in heatmap plotting. |

### Visualization
| Script | Description |
|--------|-------------|
| `plotting_scripts/plot_category_enrichment_heatmap.py` | Heatmaps of functional category fold enrichment × region, for strengthening and weakening. Args: `--log`, `--clip`, `--sig_only`, `--split_colors` |
| `plotting_scripts/plot_category_heatmap_broad_clustered.py` | Per-category LR pair heatmaps: rows = LR pairs, columns = broad sender→receiver, columns clustered by correlation. Only plots significant category × region × direction combos. |
| `plotting_scripts/plot_chord_category.py` | Chord plots for any functional category. Args: `--category`, `--grid`, `--region`, `--colorbar` |
| `plotting_scripts/plot_hypergeometric_heatmaps.py` | Heatmaps of fold enrichment by cell type × region. Args: `--log`, `--clip`, `--sig_only`, `--split_colors` |
| `plotting_scripts/plot_regression_region_celltype_heatmap.py` | Heatmap: mean age_coef per cell type pair × region |
| `plotting_scripts/plot_regression_per_region_dotplot.py` | Per-region dot plot from regression outputs |

## Output Structure
```
results/
├── per_animal_louvain_corrected/           # Per-animal LIANA outputs (corrected resource)
├── lr_matrices/                            # LR means matrices per region
└── within_region_analysis_corrected/
    ├── regression_results/
    │   ├── regression_{Region}/
    │   │   └── whole_{region}_age_sex_regression.csv
    │   └── chord_plots_{category}/         # Chord plots per functional category
    ├── hypergeometric_all_regions/         # Cell type enrichment
    │   └── hypergeometric_enrichment_all_regions.csv
    └── hypergeometric_category/            # Functional category enrichment
        ├── hypergeometric_category_enrichment_all_regions.csv
        ├── category_tables/                # Per-category LR pair CSVs
        │   └── category_{Category}.csv
        └── heatmaps_broad_clustered/       # LR pair heatmaps per category
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
- `region`, `cell_type`, `role` (`sender`/`receiver`)
- `n_tested`, `k_pos`, `expected_pos`, `fold_enrichment_pos`
- `k_neg`, `expected_neg`, `fold_enrichment_neg`
- `p_pos_enrich`, `p_pos_deplete`, `p_neg_enrich`, `p_neg_deplete`
- `q_pos_enrich`, `q_pos_deplete`, `q_neg_enrich`, `q_neg_deplete`

`hypergeometric_category_enrichment_all_regions.csv` columns:
- `region`, `category`, `n_category`
- `k_pos`, `expected_pos`, `fold_enrichment_pos`
- `k_neg`, `expected_neg`, `fold_enrichment_neg`
- `p_pos_enrich`, `p_pos_deplete`, `p_neg_enrich`, `p_neg_deplete`
- `q_pos_enrich`, `q_pos_deplete`, `q_neg_enrich`, `q_neg_deplete`

## Functional LR Categories

21 broad categories covering 2,909 LR pairs. Classification source tracked per pair (`cpdb_uniprot`, `cpdb_special`, `manual`).

| Broad category | n | All LR pair types | Function |
|---|---|---|---|
| Wnt signaling | 492 | WNT (478), R-spondin (14) | Cell fate, proliferation, synaptic plasticity via β-catenin and non-canonical pathways |
| Glutamate signaling | 400 | Glutamate biosynthetic/transport complexes (400) | Primary excitatory neurotransmission; biosynthesis, vesicular loading, reuptake |
| Immune/Inflammatory | 323 | Chemokines (119), Interleukin (68), TNF (37), HLA (23), Interferon (20), Complement (12), FC receptor (12), NK/Killer cell (11), Galectin (4), Poliovirus receptor (4), Selectin (4), CSF (3), Lymphotactin (2), Oncostatin-M (2), Pro-MHC (2) | Neuroinflammation, microglial activation, immune surveillance, cytokine signaling |
| ECM/Adhesion | 285 | Collagen/Integrin (178), Fibronectin (20), ICAM (13), CEAM (10), Cadherin (9), Nectin (7), Desmosome (6), Laminin (6), Integrin (6), JAM (5), tenascin (5), VCAM (5), THY1 (5), Podocalyxin (4), Osteopontin (3), Vitronectin (3) | Structural cell-matrix and cell-cell adhesion; tissue integrity and non-synaptic contact |
| Growth Factor/Neurotrophic | 252 | BMP (53), FGF (30), GDF (29), Inhibin/Activin (28), VEGF (13), TGFβ (13), Neurotrophin (11), IGF (10), EGF (8), Neuregulin (7), PDGF (7), NODAL (6), RET (6), Pro-neuregulin (5), Placenta GF (4), AMH (3), Betacellulin (3), BDNF (3), NGF (3), Midkine (2), Left-right determination (2), Pleiotrophin (2), Reelin (2), Sclerostin (2) | Neuronal survival, proliferation, differentiation, myelination; RTK and Smad signaling |
| GABA signaling | 238 | GABA biosynthetic/transport complexes (238) | Primary inhibitory neurotransmission; biosynthesis, vesicular loading, reuptake |
| Monoamine signaling | 177 | Serotonin (78), Serotonin/Dopamine (39), Adrenaline (27), Noradrenaline (27), Histamine (4), Melatonin (2) | Modulatory neurotransmission; mood, arousal, reward, autonomic function |
| Neuropeptide signaling | 145 | Neuropeptide precursor (13), Somatostatin (12), Galanin (12), POMC (8), Peptide YY (8), Opioid (8), Urocortin/Urotensin (7), Tachykinins (6), Gastrin (6), Neuromedin (6), Agouti-related (5), Adrenomedullin (4), VIP (4), TAFA (4), Prokineticin (4), Oxytocin (4), Calcitonin (4), Amylin (4), Vasopressin (4), Pituitary adenylate (3), Pro-glucagon (3), Neurotensin (3), Secretin (3), Orexin (2), CRF (2), Spexin (2), CCK (2), Apelin (2) | Slow neuromodulation via GPCRs; pain, stress, social behavior, homeostasis |
| Axon guidance | 98 | Ephrin (55), Semaphorin (22), L1CAM (9), Netrin (9), SLIT (3) | Directional axon navigation, fasciculation, circuit wiring via repulsive/attractive cues |
| Steroid/Retinoid signaling | 89 | Retinoic acid (34), Steroids (23), DHEA (9), Progesterone (9), Cholesterol/Desmosterol (6), Cholesterol (4), Estradiol (4) | Nuclear receptor-mediated transcriptional regulation via steroids, retinoids, and cholesterol derivatives |
| Synaptic adhesion | 71 | Neuroligin (16), LRRTM (13), LRFN (10), Calsyntenin (9), SLITRK (9), Neurexin (8), CADM (6) | Trans-synaptic adhesion complexes specifying and maintaining excitatory/inhibitory synapses |
| Metabolic/Other | 65 | Apolipoprotein (9), Transferrin (8), Amyloid-β precursor (7), Plasminogen activator (6), Prothrombin (9), Fibrinogen (3), Amyloid-like (3), Annexin (3), Growth arrest (3), Syndecan (3), von Willebrand factor (3), Phospholipase (2), Cathepsin (2), Humanin (2), Prosaposin (2) | Diverse metabolic, proteolytic, and stress-response intercellular signals |
| Hormones/Endocrine | 61 | Choriogonadotropin (11), Relaxin (10), PTH (7), Endothelin (6), GH-releasing hormone (4), GnRH (4), Angiopoietin (3), Angiotensin (3), Insulin-like precursor (3), Kininogen (3), Leptin (3), Prolactin (2), T3 (2) | Peptide and protein hormone signaling; neuroendocrine and systemic homeostatic regulation |
| Lipid signaling | 60 | Lipoxin/Leukotriene (26), Prostaglandin (22), Arachidonoylglycerol (6), LPA receptor (3), Thromboxane (3) | Eicosanoid and lysophospholipid signaling; inflammation, pain, and neuromodulation via GPCRs |
| Notch/Hedgehog | 46 | Notch (31), Sonic hedgehog (7), Desert hedgehog (4), Indian hedgehog (4) | Cell fate determination, glial differentiation, adult neural stem cell maintenance |
| Cholinergic signaling | 43 | Choline acetyltransferase/CHAT complexes (43) | CNS and neuromuscular cholinergic transmission; attention, memory, autonomic function |
| Purinergic signaling | 20 | Adenosine/NT5E complexes (16), Ectonucleoside (4) | Adenosine production and transport; synaptic depression and neuroinflammation modulation |
| Teneurin/Latrophilin | 19 | Teneurin/FLRT complexes (19) | Trans-synaptic partner matching during circuit assembly; synapse specificity |
| Gap junction | 10 | Connexin gap junctions (10) | Direct intercellular electrical and metabolic coupling via hemichannel complexes |
| Glycine signaling | 8 | Glycine biosynthetic/transport complexes (8) | Inhibitory neurotransmission in brainstem/spinal cord; glycine biosynthesis and reuptake |
| Synapse organization | 7 | Thrombospondin (5), Agrin (2) | ECM-derived inductive signals triggering synapse formation and maturation |

## Key Parameters

- **CCC**: `expr_prop=0.05`, FDR correction on `cellphone_pvals`, min 1 animal per interaction
- **Regression**: `min_animals=10`, `min_age=1.0` (excludes infants)
- **Enrichment**: `age_qval < 0.05` (regression significance threshold), hypergeometric test with global BH-FDR correction

## Environment
```bash
conda activate cellchat_env    # CCC pipeline (Python)
```
