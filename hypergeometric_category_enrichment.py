#!/usr/bin/env python3
"""
Hypergeometric enrichment AND depletion of functional LR categories per region,
separately for strengthening (pos beta) and weakening (neg beta) age effects.
No cell type stratification — tests whether each functional category is over/under-
represented among age-significant interactions in each region.

Hypergeometric framing (per region, per direction, per category):
  N = total tested rows in region
  K = total age-sig rows with pos (or neg) beta
  n = rows in region where LR pair belongs to category Y
  k = rows in region where LR pair belongs to category Y AND age-sig pos (or neg)

  Enrichment p-value = P(X > k)  = hypergeom.sf(k, N, K, n)
  Depletion  p-value = P(X <= k) = hypergeom.cdf(k, N, K, n)

Single global BH-FDR correction across all tests.

Usage:
  python hypergeometric_category_enrichment.py
"""

import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
REGRESSION_DIR  = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_results"
ANNOTATIONS     = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
OUT_DIR         = os.path.join(REGRESSION_DIR, "hypergeometric_all_regions")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH = 0.05

regression_files = sorted(glob.glob(
    os.path.join(REGRESSION_DIR, "regression_*/whole_*_age_sex_regression.csv")
))
print(f"Found {len(regression_files)} region files:")
for f in regression_files:
    print(f"  {os.path.basename(f)}")

# ── load annotations ───────────────────────────────────────────────────────
ann = pd.read_csv(ANNOTATIONS)
ann = ann[["lr_pair", "broad_category"]].drop_duplicates()
categories = sorted(ann["broad_category"].unique())
print(f"\nLoaded {len(ann)} LR pair annotations across {len(categories)} categories")

# ── hypergeometric test function ───────────────────────────────────────────
def run_hypergeom(N, K, n, k):
    expected  = n * K / N if N > 0 else np.nan
    fold      = k / expected if expected > 0 else np.nan
    p_enrich  = hypergeom.sf(k, N, K, n)
    p_deplete = hypergeom.cdf(k, N, K, n)
    return k, expected, fold, p_enrich, p_deplete

# ── main loop ──────────────────────────────────────────────────────────────
all_results = []

for fpath in regression_files:
    basename = os.path.basename(fpath)
    region = basename.replace("whole_", "").replace("_age_sex_regression.csv", "")

    print(f"\n{'='*60}")
    print(f"Processing: {region}")
    print(f"{'='*60}")

    df = pd.read_csv(fpath)
    print(f"  Loaded {len(df):,} rows")

    # parse ligand and receptor, build lr_pair key
    df[["sender", "receiver", "ligand", "receptor"]] = df["interaction"].str.split("|", expand=True)
    df["lr_pair"] = df["ligand"] + "|" + df["receptor"]

    # merge annotations
    df = df.merge(ann, on="lr_pair", how="left")
    n_unmapped = df["broad_category"].isna().sum()
    if n_unmapped > 0:
        print(f"  WARNING: {n_unmapped:,} rows could not be mapped to a category")

    df_sig     = df[df["age_qval"] < Q_THRESH]
    df_sig_pos = df_sig[df_sig["age_coef"] > 0]
    df_sig_neg = df_sig[df_sig["age_coef"] < 0]

    N     = len(df)
    K_pos = len(df_sig_pos)
    K_neg = len(df_sig_neg)

    print(f"  N={N:,}  K_pos={K_pos}  K_neg={K_neg}")

    if K_pos + K_neg == 0:
        print(f"  WARNING: no age-significant interactions in {region}, skipping.")
        continue

    rows = []
    for cat in categories:
        cat_mask     = df["broad_category"] == cat
        cat_pos_mask = df_sig_pos["broad_category"] == cat
        cat_neg_mask = df_sig_neg["broad_category"] == cat

        n_cat = cat_mask.sum()
        k_pos = cat_pos_mask.sum()
        k_neg = cat_neg_mask.sum()

        k_pos_val, exp_pos, fe_pos, p_pos_enrich, p_pos_deplete = run_hypergeom(N, K_pos, n_cat, k_pos)
        k_neg_val, exp_neg, fe_neg, p_neg_enrich, p_neg_deplete = run_hypergeom(N, K_neg, n_cat, k_neg)

        rows.append({
            "region":               region,
            "category":             cat,
            "n_category":           n_cat,
            "k_pos":                k_pos_val,
            "expected_pos":         round(exp_pos, 2),
            "fold_enrichment_pos":  round(fe_pos, 3),
            "p_pos_enrich":         p_pos_enrich,
            "p_pos_deplete":        p_pos_deplete,
            "k_neg":                k_neg_val,
            "expected_neg":         round(exp_neg, 2),
            "fold_enrichment_neg":  round(fe_neg, 3),
            "p_neg_enrich":         p_neg_enrich,
            "p_neg_deplete":        p_neg_deplete,
        })

    all_results.append(pd.DataFrame(rows))

# ── combine and apply single global FDR correction ────────────────────────
combined = pd.concat(all_results, ignore_index=True)

all_pvals = pd.concat([
    combined["p_pos_enrich"],
    combined["p_pos_deplete"],
    combined["p_neg_enrich"],
    combined["p_neg_deplete"],
], ignore_index=True)

print(f"\nRunning global FDR correction over {len(all_pvals):,} p-values "
      f"({len(combined):,} each for pos_enrich, pos_deplete, neg_enrich, neg_deplete)...")
_, all_qvals, _, _ = multipletests(all_pvals, method="fdr_bh")
n = len(combined)
combined["q_pos_enrich"]  = all_qvals[0*n:1*n]
combined["q_pos_deplete"] = all_qvals[1*n:2*n]
combined["q_neg_enrich"]  = all_qvals[2*n:3*n]
combined["q_neg_deplete"] = all_qvals[3*n:4*n]

# ── save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "hypergeometric_category_enrichment_all_regions.csv")
combined.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Total rows: {len(combined)}")

# ── print summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CROSS-REGION SUMMARY (q<0.05):")
print("="*60)
for label, qcol, kcol, ecol, fecol in [
    ("strengthening — enriched",  "q_pos_enrich",  "k_pos", "expected_pos", "fold_enrichment_pos"),
    ("strengthening — depleted",  "q_pos_deplete", "k_pos", "expected_pos", "fold_enrichment_pos"),
    ("weakening — enriched",      "q_neg_enrich",  "k_neg", "expected_neg", "fold_enrichment_neg"),
    ("weakening — depleted",      "q_neg_deplete", "k_neg", "expected_neg", "fold_enrichment_neg"),
]:
    sig = combined[combined[qcol] < Q_THRESH].sort_values(["category", "region"])
    print(f"\n  {label.upper()}:")
    if len(sig):
        print(sig[["region", "category", "n_category", kcol, ecol, fecol, qcol]].to_string(index=False))
    else:
        print("    none")
