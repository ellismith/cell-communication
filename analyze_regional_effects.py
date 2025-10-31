#!/usr/bin/env python3
"""
analyze_regional_effects.py
----------------------------
Combine all *_w_region_results.csv and compute regional differences
for each ligand–receptor interaction using one-way ANOVA.
Outputs:
- regional_effects_detailed.csv  (one row per LR per region)
- regional_ANOVA_results.csv     (p-values, FDR)
"""
import os, glob
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

base = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = "/scratch/easmit31/cell_cell/results/"
os.makedirs(out_dir, exist_ok=True)

files = glob.glob(os.path.join(base, "*_w_region_results.csv"))
if not files:
    raise FileNotFoundError("No *_w_region_results.csv found!")

print(f"Found {len(files)} region-aware result files")

all_rows = []
for f in files:
    pair_name = os.path.basename(f).replace("_w_region_results.csv", "")
    df = pd.read_csv(f)
    if not {"source","target","ligand_complex","receptor_complex","lr_means","region"} <= set(df.columns):
        continue
    df["interaction"] = (
        df["source"] + "→" + df["target"] + ":" +
        df["ligand_complex"] + "→" + df["receptor_complex"]
    )
    df["pair_name"] = pair_name
    all_rows.append(df[["pair_name","interaction","region","lr_means"]])

data = pd.concat(all_rows, ignore_index=True)
data.to_csv(os.path.join(out_dir, "regional_effects_detailed.csv"), index=False)
print(f"✓ Saved detailed data to regional_effects_detailed.csv")

# Run ANOVA per interaction
results = []
for inter, sub in data.groupby("interaction"):
    if sub["region"].nunique() > 1:
        model = ols("lr_means ~ C(region)", data=sub).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p = anova["PR(>F)"].iloc[0]
        results.append({"interaction": inter, "p_region": p})

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    res_df["padj"] = sm.stats.multitest.multipletests(res_df["p_region"], method="fdr_bh")[1]
    res_df.sort_values("padj", inplace=True)
    res_df.to_csv(os.path.join(out_dir, "regional_ANOVA_results.csv"), index=False)
    print(f"✓ Saved ANOVA summary to regional_ANOVA_results.csv")
else:
    print("⚠️ No interactions with multiple regions; skipping ANOVA")
