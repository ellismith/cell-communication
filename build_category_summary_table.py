#!/usr/bin/env python3
"""
Builds a summary table connecting category enrichment results to cell type
and LR pair level detail.

For each significant category × region × direction (q<0.05), shows:
  - fold_enrichment, q_val
  - top broad sender→receiver combo (by n Louvain combos)
  - n_pos_combos, n_neg_combos, pct_agreement for that combo
  - top 3 LR pairs by n Louvain combos

Output: category_summary_table.csv
"""

import pandas as pd
import numpy as np
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR   = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
ENRICH_FILE    = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/hypergeometric_category_enrichment_all_regions.csv"
OUT_DIR        = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions"

Q_THRESH = 0.05

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

# ── load enrichment results ────────────────────────────────────────────────
enrich = pd.read_csv(ENRICH_FILE)

# ── load all category tables ───────────────────────────────────────────────
category_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
cat_data = {}
for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$", "", regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$", "", regex=True)
    cat_data[cat_name] = df

# ── build summary rows ─────────────────────────────────────────────────────
rows = []

for _, erow in enrich.iterrows():
    region   = erow["region"]
    category = erow["category"]

    if category not in cat_data:
        continue

    df = cat_data[category]
    df_region = df[df["region"] == region]
    if len(df_region) == 0:
        continue

    for direction, qcol, fecol in [
        ("strengthening", "q_pos_enrich", "fold_enrichment_pos"),
        ("weakening",     "q_neg_enrich", "fold_enrichment_neg"),
    ]:
        q_val = erow[qcol]
        if q_val >= Q_THRESH:
            continue

        fe = erow[fecol]
        df_dir = df_region  # include all combos to compute agreement

        # aggregate by broad sender→receiver
        agg = df_dir.groupby(["sender_broad", "receiver_broad"]).apply(
            lambda x: pd.Series({
                "n_pos": (x["direction"] == "strengthening").sum(),
                "n_neg": (x["direction"] == "weakening").sum(),
                "n_total": len(x),
            })
        ).reset_index()

        agg["pct_agreement"] = agg.apply(
            lambda r: max(r["n_pos"], r["n_neg"]) / r["n_total"] * 100
            if r["n_total"] > 0 else np.nan, axis=1
        )
        agg["dominant_direction"] = agg.apply(
            lambda r: "strengthening" if r["n_pos"] >= r["n_neg"] else "weakening", axis=1
        )

        # filter to combos where dominant direction matches
        agg_dir = agg[agg["dominant_direction"] == direction].sort_values(
            "n_total", ascending=False)

        if len(agg_dir) == 0:
            # no combos in this direction, still report enrichment
            top_sender = top_receiver = ""
            n_pos = n_neg = n_total = 0
            pct_agreement = np.nan
        else:
            top = agg_dir.iloc[0]
            top_sender    = top["sender_broad"]
            top_receiver  = top["receiver_broad"]
            n_pos         = int(top["n_pos"])
            n_neg         = int(top["n_neg"])
            n_total       = int(top["n_total"])
            pct_agreement = round(top["pct_agreement"], 1)

        # top 3 LR pairs by n combos in dominant direction
        df_dir_only = df_region[df_region["direction"] == direction]
        top_lrs = (
            df_dir_only.groupby("lr_pair").size()
            .sort_values(ascending=False)
            .head(3).index.tolist()
        )

        rows.append({
            "region":          REGION_LABELS.get(region, region),
            "category":        category,
            "direction":       direction,
            "fold_enrichment": round(fe, 3),
            "q_val":           q_val,
            "top_sender":      top_sender,
            "top_receiver":    top_receiver,
            "n_pos_combos":    n_pos,
            "n_neg_combos":    n_neg,
            "n_total_combos":  n_total,
            "pct_agreement":   pct_agreement,
            "top_lr_pairs":    " | ".join(top_lrs),
        })

# ── save ───────────────────────────────────────────────────────────────────
summary = pd.DataFrame(rows).sort_values(
    ["direction", "category", "region"])

out_path = os.path.join(OUT_DIR, "category_summary_table.csv")
summary.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Total rows: {len(summary)}")

# ── print ──────────────────────────────────────────────────────────────────
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", 200)
print("\n" + "="*80)
print("STRENGTHENING:")
print("="*80)
print(summary[summary["direction"]=="strengthening"][
    ["region","category","fold_enrichment","q_val","top_sender","top_receiver",
     "n_pos_combos","n_neg_combos","pct_agreement","top_lr_pairs"]
].to_string(index=False))

print("\n" + "="*80)
print("WEAKENING:")
print("="*80)
print(summary[summary["direction"]=="weakening"][
    ["region","category","fold_enrichment","q_val","top_sender","top_receiver",
     "n_pos_combos","n_neg_combos","pct_agreement","top_lr_pairs"]
].to_string(index=False))
