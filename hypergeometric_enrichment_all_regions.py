#!/usr/bin/env python3
"""
Hypergeometric enrichment AND depletion test for age-associated LR pairs, all 11 regions.
Directional only: separately tests strengthening (pos beta) and weakening (neg beta).
FDR correction is applied once globally across all regions, cell types, roles,
directions, and enrichment/depletion tests combined.

Hypergeometric framing (per region, per direction):
  N = total tested rows in region
  K = total age-sig rows with pos (or neg) beta
  n = rows where cell type X is sender (or receiver)
  k = rows where cell type X is sender AND age-sig AND pos (or neg) beta

  Enrichment p-value = P(X > k)  = hypergeom.sf(k, N, K, n)
  Depletion  p-value = P(X <= k) = hypergeom.cdf(k, N, K, n)
"""

import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
REGRESSION_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis"
OUT_DIR        = os.path.join(REGRESSION_DIR, "hypergeometric_all_regions")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH = 0.05

regression_files = sorted(glob.glob(
    os.path.join(REGRESSION_DIR, "regression_*/whole_*_age_sex_regression.csv")
))
print(f"Found {len(regression_files)} region files:")
for f in regression_files:
    print(f"  {os.path.basename(f)}")

# ── hypergeometric test function ───────────────────────────────────────────
def run_hypergeom(N, K, n, k):
    expected   = n * K / N if N > 0 else np.nan
    fold       = k / expected if expected > 0 else np.nan
    p_enrich   = hypergeom.sf(k, N, K, n)
    p_deplete  = hypergeom.cdf(k, N, K, n)
    return k, expected, fold, p_enrich, p_deplete

# ── run tests for one role ─────────────────────────────────────────────────
def enrichment_by_role(df_all, df_sig_pos, df_sig_neg, N, K_pos, K_neg, role):
    cell_types = sorted(df_all[role].unique())
    rows = []
    for ct in cell_types:
        n = (df_all[role] == ct).sum()

        k_pos, exp_pos, fe_pos, p_pos_enrich, p_pos_deplete = run_hypergeom(
            N, K_pos, n, (df_sig_pos[role] == ct).sum())
        k_neg, exp_neg, fe_neg, p_neg_enrich, p_neg_deplete = run_hypergeom(
            N, K_neg, n, (df_sig_neg[role] == ct).sum())

        rows.append({
            "cell_type":            ct,
            "role":                 role,
            "n_tested":             n,
            "k_pos":                k_pos,
            "expected_pos":         round(exp_pos, 2),
            "fold_enrichment_pos":  round(fe_pos, 3),
            "p_pos_enrich":         p_pos_enrich,
            "p_pos_deplete":        p_pos_deplete,
            "k_neg":                k_neg,
            "expected_neg":         round(exp_neg, 2),
            "fold_enrichment_neg":  round(fe_neg, 3),
            "p_neg_enrich":         p_neg_enrich,
            "p_neg_deplete":        p_neg_deplete,
        })
    return pd.DataFrame(rows)

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

    df[["sender", "receiver", "ligand", "receptor"]] = df["interaction"].str.split("|", expand=True)
    df["sender"]   = df["sender"].str.replace(r"_\d+$", "", regex=True)
    df["receiver"] = df["receiver"].str.replace(r"_\d+$", "", regex=True)

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

    results = []
    for role in ["sender", "receiver"]:
        res = enrichment_by_role(df, df_sig_pos, df_sig_neg, N, K_pos, K_neg, role)
        results.append(res)
    results = pd.concat(results, ignore_index=True)
    results.insert(0, "region", region)
    all_results.append(results)

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
out_path = os.path.join(OUT_DIR, "hypergeometric_enrichment_all_regions.csv")
combined.to_csv(out_path, index=False)
print(f"\nSaved combined results: {out_path}")
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
    sig = combined[combined[qcol] < Q_THRESH].sort_values(["role", "cell_type", "region"])
    print(f"\n  {label.upper()}:")
    if len(sig):
        print(sig[["region", "role", "cell_type", kcol, ecol, fecol, qcol]].to_string(index=False))
    else:
        print("    none")
