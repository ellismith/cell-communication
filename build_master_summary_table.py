#!/usr/bin/env python3
"""
Master summary table: one row per significant category × region × direction.
Shows top sender→receiver combo and Louvain direction agreement.
"""

import pandas as pd
import numpy as np
import os
import glob

BASE_DIR        = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
HYPERGEOM_BROAD = os.path.join(BASE_DIR, "hypergeometric_all_regions", "hypergeometric_enrichment_all_regions.csv")
HYPERGEOM_CAT   = os.path.join(BASE_DIR, "hypergeometric_all_regions", "hypergeometric_category_enrichment_all_regions.csv")
CATEGORY_DIR    = os.path.join(BASE_DIR, "hypergeometric_all_regions", "category_tables")
OUT_DIR         = os.path.join(BASE_DIR, "hypergeometric_all_regions")

Q_THRESH = 0.05

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

print("Loading data...")
broad      = pd.read_csv(HYPERGEOM_BROAD)
cat_enrich = pd.read_csv(HYPERGEOM_CAT)

cat_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
cat_dfs = []
for f in cat_files:
    cat_name = os.path.basename(f).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(f)
    df["category"] = cat_name
    cat_dfs.append(df)
cat_all = pd.concat(cat_dfs, ignore_index=True)
cat_all["sender_broad"]   = cat_all["sender"].str.replace(r"_\d+$", "", regex=True)
cat_all["receiver_broad"] = cat_all["receiver"].str.replace(r"_\d+$", "", regex=True)

# sender/receiver q-value lookup
sender_q   = {(r["region"], r["cell_type"]): (r["q_pos_enrich"], r["q_neg_enrich"])
              for _, r in broad[broad["role"] == "sender"].iterrows()}
receiver_q = {(r["region"], r["cell_type"]): (r["q_pos_enrich"], r["q_neg_enrich"])
              for _, r in broad[broad["role"] == "receiver"].iterrows()}

rows = []
for _, erow in cat_enrich.iterrows():
    region   = erow["region"]
    category = erow["category"]

    for direction, qcol, fecol in [
        ("strengthening", "q_pos_enrich", "fold_enrichment_pos"),
        ("weakening",     "q_neg_enrich", "fold_enrichment_neg"),
    ]:
        q_val = erow[qcol]
        if q_val >= Q_THRESH:
            continue
        fe = erow[fecol]

        sub = cat_all[(cat_all["category"] == category) & (cat_all["region"] == region)]
        if len(sub) == 0:
            continue

        # overall Louvain agreement across ALL combos in this category × region
        n_agree    = (sub["direction"] == direction).sum()
        n_disagree = (sub["direction"] != direction).sum()
        n_total    = len(sub)
        pct_agree  = round(n_agree / n_total * 100, 1) if n_total > 0 else np.nan

        # top sender→receiver combo (most Louvain combos in agreed direction)
        sub_dir = sub[sub["direction"] == direction]
        if len(sub_dir) == 0:
            continue
        top_combo = (
            sub_dir.groupby(["sender_broad", "receiver_broad"])
            .size().sort_values(ascending=False).reset_index(name="n").iloc[0]
        )
        top_sender   = top_combo["sender_broad"]
        top_receiver = top_combo["receiver_broad"]

        # q-values for top sender and receiver
        sq = sender_q.get((region, top_sender), (np.nan, np.nan))
        rq = receiver_q.get((region, top_receiver), (np.nan, np.nan))
        s_q = sq[0] if direction == "strengthening" else sq[1]
        r_q = rq[0] if direction == "strengthening" else rq[1]

        # top 5 LR pairs overall in this direction
        top_lrs = sub_dir["lr_pair"].value_counts().head(5).index.tolist()
        n_unique_lrs = sub_dir["lr_pair"].nunique()

        rows.append({
            "region":               REGION_LABELS.get(region, region),
            "category":             category,
            "direction":            direction,
            "fold_enrichment":      round(fe, 3),
            "q_val_category":       q_val,
            "n_louvain_agree":      n_agree,
            "n_louvain_disagree":   n_disagree,
            "pct_louvain_agree":    pct_agree,
            "top_sender":           top_sender,
            "top_receiver":         top_receiver,
            "q_val_sender":         s_q,
            "q_val_receiver":       r_q,
            "n_unique_lr_pairs":    n_unique_lrs,
            "top_5_lr_pairs":       " | ".join(top_lrs),
        })

master = pd.DataFrame(rows).sort_values(["direction", "q_val_category"])

out_path = os.path.join(OUT_DIR, "master_summary_table.csv")
master.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(master)} rows)")

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 250)
for direction in ["strengthening", "weakening"]:
    print(f"\n{'='*80}\n{direction.upper()}\n{'='*80}")
    sub = master[master["direction"] == direction]
    print(sub[["region","category","fold_enrichment","q_val_category",
               "n_louvain_agree","n_louvain_disagree","pct_louvain_agree",
               "top_sender","top_receiver","q_val_sender","q_val_receiver",
               "n_unique_lr_pairs","top_5_lr_pairs"]].to_string(index=False))
