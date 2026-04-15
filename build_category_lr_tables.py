#!/usr/bin/env python3
"""
For each functional category that is significantly enriched or depleted
(q<0.05) in at least one region × direction combination, output a CSV
with all age-significant LR pairs in that category, their sender→receiver
Louvain breakdowns, age coefficients, and direction.
Output: one CSV per significant category in:
  results/within_region_analysis_corrected/hypergeometric_category/category_tables/
"""
import pandas as pd
import numpy as np
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
REGRESSION_DIR  = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_results"
ANNOTATIONS     = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
CATEGORY_ENRICH = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/hypergeometric_category/hypergeometric_category_enrichment_all_regions.csv"
OUT_DIR         = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/hypergeometric_category/category_tables"
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH = 0.05

# ── load ───────────────────────────────────────────────────────────────────
enrich = pd.read_csv(CATEGORY_ENRICH)
ann    = pd.read_csv(ANNOTATIONS)[["lr_pair", "broad_category"]].drop_duplicates()

# significant categories across any region × direction
sig_cats = set()
for qcol in ["q_pos_enrich", "q_neg_enrich"]:
    sig_cats |= set(enrich[enrich[qcol] < Q_THRESH]["category"].unique())
print(f"Significant categories: {len(sig_cats)}")
print(sorted(sig_cats))

# ── load regression files ──────────────────────────────────────────────────
regression_files = sorted(glob.glob(
    os.path.join(REGRESSION_DIR, "regression_*/whole_*_age_sex_regression.csv")
))
print(f"\nFound {len(regression_files)} regression files")

all_sig = []
for fpath in regression_files:
    basename = os.path.basename(fpath)
    region = basename.replace("whole_", "").replace("_age_sex_regression.csv", "")
    print(f"  Loading {region}...")

    chunks = []
    for chunk in pd.read_csv(fpath, chunksize=500000):
        chunk[["sender","receiver","ligand","receptor"]] = chunk["interaction"].str.split("|", expand=True)
        chunk["lr_pair"] = chunk["ligand"] + "|" + chunk["receptor"]
        sub = chunk[chunk["age_qval"] < Q_THRESH].copy()
        if len(sub):
            chunks.append(sub)

    if not chunks:
        continue

    sig = pd.concat(chunks)
    sig = sig.merge(ann, on="lr_pair", how="left")
    sig["region"]    = region
    sig["direction"] = np.where(sig["age_coef"] > 0, "strengthening", "weakening")
    all_sig.append(sig[["region","sender","receiver","lr_pair","broad_category",
                         "age_coef","age_qval","direction"]])

combined = pd.concat(all_sig, ignore_index=True)
print(f"\nTotal significant interactions: {len(combined):,}")

# ── save one CSV per significant category ─────────────────────────────────
for cat in sig_cats:
    sub = combined[combined["broad_category"] == cat]
    if len(sub) == 0:
        continue
    fname = f"category_{cat.replace(' ','_').replace('/','_')}.csv"
    fpath = os.path.join(OUT_DIR, fname)
    sub.to_csv(fpath, index=False)
    print(f"  Saved: {fname}  ({len(sub):,} rows)")

print("\nDone.")
