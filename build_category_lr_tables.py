#!/usr/bin/env python3
"""
For each functional category that is significantly enriched or depleted
(q<0.05) in at least one region × direction combination, output a CSV
with all age-significant LR pairs in that category, their sender→receiver
Louvain breakdowns, age coefficients, and direction.

Output: one CSV per significant category in:
  results/within_region_analysis_corrected/hypergeometric_all_regions/category_tables/
"""

import pandas as pd
import numpy as np
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
REGRESSION_DIR  = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
ANNOTATIONS     = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
CATEGORY_ENRICH = os.path.join(REGRESSION_DIR, "hypergeometric_all_regions",
                               "hypergeometric_category_enrichment_all_regions.csv")
OUT_DIR         = os.path.join(REGRESSION_DIR, "hypergeometric_all_regions", "category_tables")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH = 0.05

# ── load enrichment results and annotations ────────────────────────────────
enrich = pd.read_csv(CATEGORY_ENRICH)
ann    = pd.read_csv(ANNOTATIONS)[["lr_pair", "broad_category"]].drop_duplicates()

# find categories significant in at least one region × direction × enrich/deplete combo
q_cols = ["q_pos_enrich", "q_pos_deplete", "q_neg_enrich", "q_neg_deplete"]
sig_mask = (enrich[q_cols] < Q_THRESH).any(axis=1)
sig_categories = enrich.loc[sig_mask, "category"].unique()
print(f"Found {len(sig_categories)} significant categories:")
for cat in sorted(sig_categories):
    print(f"  {cat}")

# ── load all regression files ──────────────────────────────────────────────
regression_files = sorted(glob.glob(
    os.path.join(REGRESSION_DIR, "regression_*/whole_*_age_sex_regression.csv")
))

all_sig_rows = []
for fpath in regression_files:
    basename = os.path.basename(fpath)
    region = basename.replace("whole_", "").replace("_age_sex_regression.csv", "")

    df = pd.read_csv(fpath)
    df[["sender", "receiver", "ligand", "receptor"]] = df["interaction"].str.split("|", expand=True)
    df["lr_pair"] = df["ligand"] + "|" + df["receptor"]
    df = df.merge(ann, on="lr_pair", how="left")

    # keep only age-significant rows
    df_sig = df[df["age_qval"] < Q_THRESH].copy()
    df_sig["direction"] = np.where(df_sig["age_coef"] > 0, "strengthening", "weakening")
    df_sig["region"] = region

    all_sig_rows.append(df_sig[["region", "sender", "receiver", "lr_pair",
                                 "broad_category", "age_coef", "age_qval",
                                 "direction", "n_animals", "mean_lr_means"]])

all_sig = pd.concat(all_sig_rows, ignore_index=True)
print(f"\nTotal age-significant rows across all regions: {len(all_sig):,}")

# ── write one CSV per significant category ─────────────────────────────────
for cat in sorted(sig_categories):
    cat_df = all_sig[all_sig["broad_category"] == cat].copy()
    cat_df = cat_df.sort_values(["region", "direction", "sender", "receiver", "lr_pair"])

    # clean filename
    fname = "category_" + cat.replace("/", "_").replace(" ", "_") + ".csv"
    fpath = os.path.join(OUT_DIR, fname)
    cat_df.to_csv(fpath, index=False)
    print(f"Saved: {fname}  ({len(cat_df):,} rows)")

print(f"\nDone. All category tables saved to: {OUT_DIR}")
