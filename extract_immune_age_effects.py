#!/usr/bin/env python3
import pandas as pd
import os

ANN = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
REG_BASE = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_results"
OUT = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/immune_age_significant.csv"

ann = pd.read_csv(ANN)
immune_pairs = set(ann[ann["broad_category"] == "Immune/Inflammatory"]["lr_pair"])
print(f"Total immune LR pairs in annotations: {len(immune_pairs)}")

region_map = {
    "ACC":"acc","CN":"cn","dlPFC":"dlpfc","EC":"ec","HIP":"hip",
    "IPP":"ipp","lCb":"lcb","M1":"m1","MB":"mb","mdTN":"mdtn","NAc":"nac"
}

rows = []
for reg, reg_lower in region_map.items():
    fpath = os.path.join(REG_BASE, f"regression_{reg}", f"whole_{reg_lower}_age_sex_regression.csv")
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}")
        continue
    df = pd.read_csv(fpath)
    df[["sender","receiver","ligand","receptor"]] = df["interaction"].str.split("|", expand=True)
    df["lr_pair"] = df["ligand"] + "|" + df["receptor"]
    df_immune = df[df["lr_pair"].isin(immune_pairs)]
    df_sig = df_immune[df_immune["age_qval"] < 0.05].copy()
    df_sig["region"] = reg
    print(f"{reg}: {len(df_immune)} immune tested, {len(df_sig)} age q<0.05")
    rows.append(df_sig)

if rows:
    out = pd.concat(rows).sort_values(["region","age_qval"])
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} rows to {OUT}")
    print(out[["region","interaction","age_coef","age_qval"]].to_string())
else:
    print("No results found.")
