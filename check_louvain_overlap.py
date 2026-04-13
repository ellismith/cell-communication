#!/usr/bin/env python3
"""
Checks overlap between:
1. Louvain subclusters significant in hypergeometric enrichment (q<0.05)
2. Louvain subclusters with age-sig LR pairs in raw regression (age_qval<0.05)

For each significant broad cell type × region × direction.

Output: louvain_overlap_check.csv
"""

import pandas as pd
import numpy as np
import glob
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
BROAD_FILE = os.path.join(BASE_DIR, "hypergeometric_all_regions",
                          "hypergeometric_enrichment_all_regions.csv")
LOU_FILE   = os.path.join(BASE_DIR, "hypergeometric_all_regions",
                          "hypergeometric_enrichment_all_regions_louvain.csv")
OUT_DIR    = os.path.join(BASE_DIR, "hypergeometric_all_regions")

Q_THRESH = 0.05

# ── load ───────────────────────────────────────────────────────────────────
print("Loading broad and Louvain enrichment files...")
broad = pd.read_csv(BROAD_FILE)
lou   = pd.read_csv(LOU_FILE)
lou["broad"] = lou["cell_type"].str.replace(r"_\d+$", "", regex=True)

print("Loading regression files...")
reg_dfs = []
for fpath in sorted(glob.glob(os.path.join(BASE_DIR,
                               "regression_*/whole_*_age_sex_regression.csv"))):
    region = os.path.basename(fpath).replace("whole_", "").replace(
              "_age_sex_regression.csv", "")
    df = pd.read_csv(fpath)
    df[["sender_lou", "receiver_lou", "ligand", "receptor"]] = \
        df["interaction"].str.split("|", expand=True)
    df["region"] = region
    reg_dfs.append(df)

reg = pd.concat(reg_dfs, ignore_index=True)
reg_sig = reg[reg["age_qval"] < Q_THRESH].copy()
reg_sig["direction"] = reg_sig["age_coef"].apply(
    lambda x: "strengthening" if x > 0 else "weakening")

print(f"  Regression: {len(reg):,} rows, {len(reg_sig):,} age-sig")

# ── main loop ──────────────────────────────────────────────────────────────
rows = []

for direction, q_col in [("strengthening", "q_pos_enrich"),
                          ("weakening",     "q_neg_enrich")]:
    broad_sig = broad[broad[q_col] < Q_THRESH][["region", "cell_type", "role"]]

    for _, brow in broad_sig.iterrows():
        region    = brow["region"]
        cell_type = brow["cell_type"]
        role      = brow["role"]
        role_col  = "sender_lou" if role == "sender" else "receiver_lou"

        # Louvain subclusters from hypergeometric
        sub_lou = lou[
            (lou["region"] == region) &
            (lou["broad"]  == cell_type) &
            (lou["role"]   == role)
        ]
        hyp_sig_set = set(sub_lou[sub_lou[q_col] < Q_THRESH]["cell_type"])
        hyp_all_set = set(sub_lou["cell_type"])

        # Louvain subclusters with age-sig LR pairs in regression
        sub_reg = reg_sig[
            (reg_sig["region"]    == region) &
            (reg_sig["direction"] == direction) &
            (reg_sig[role_col].str.replace(r"_\d+$", "", regex=True) == cell_type)
        ]
        reg_sig_set = set(sub_reg[role_col].unique())

        overlap  = hyp_sig_set & reg_sig_set
        hyp_only = hyp_sig_set - reg_sig_set
        reg_only = reg_sig_set - hyp_sig_set

        pct_overlap_of_hyp = round(len(overlap) / len(hyp_sig_set) * 100, 1) \
                             if hyp_sig_set else np.nan
        pct_overlap_of_reg = round(len(overlap) / len(reg_sig_set) * 100, 1) \
                             if reg_sig_set else np.nan

        rows.append({
            "region":               region,
            "cell_type":            cell_type,
            "role":                 role,
            "direction":            direction,
            "n_louvain_total":      len(hyp_all_set),
            "n_hyp_sig":            len(hyp_sig_set),
            "n_reg_sig":            len(reg_sig_set),
            "n_overlap":            len(overlap),
            "pct_overlap_of_hyp":   pct_overlap_of_hyp,
            "pct_overlap_of_reg":   pct_overlap_of_reg,
            "hyp_only":             ", ".join(sorted(hyp_only)),
            "reg_only":             ", ".join(sorted(reg_only)),
            "overlap":              ", ".join(sorted(overlap)),
        })

# ── save ───────────────────────────────────────────────────────────────────
result = pd.DataFrame(rows).sort_values(
    ["direction", "region", "cell_type", "role"])

out_path = os.path.join(OUT_DIR, "louvain_overlap_check.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(result)} rows)")

# ── print summary ──────────────────────────────────────────────────────────
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

for direction in ["strengthening", "weakening"]:
    print(f"\n{'='*70}")
    print(f"{direction.upper()}")
    print(f"{'='*70}")
    sub = result[result["direction"] == direction]
    print(sub[["region", "cell_type", "role",
               "n_louvain_total", "n_hyp_sig", "n_reg_sig",
               "n_overlap", "pct_overlap_of_hyp",
               "pct_overlap_of_reg"]].to_string(index=False))

print("\nDone.")
