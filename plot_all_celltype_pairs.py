#!/usr/bin/env python3
"""
plot_all_celltype_pairs.py
---------------------------
Plots each cell-type pair in both directions separately.
E.g. GABA -> Glutamatergic and Glutamatergic -> GABA.
Removes duplicates per region–interaction combination.
"""
import os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures/by_celltype_pair")
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Found {len(files)} pairwise result files")

for f in files:
    pair_name = os.path.basename(f).replace("_results.csv", "")
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        print(f"⚠️ Skipping {pair_name} (no lr_means)")
        continue

    parts = pair_name.split("_")
    mid = len(parts) // 2
    type1 = "_".join(parts[:mid])
    type2 = "_".join(parts[mid:])

    print(f"\n=== {type1} ↔ {type2} ===")

    # Parse regions
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    df_within = df[df["source_region"] == df["target_region"]].copy()
    df_within["region"] = df_within["source_region"]

    if len(df_within) == 0:
        print("⚠️ No within-region interactions")
        continue

    # Iterate over directions: type1->type2 and type2->type1
    for sender, receiver in [(type1, type2), (type2, type1)]:
        df_dir = df_within[
            (df_within["source"].str.startswith(sender)) &
            (df_within["target"].str.startswith(receiver))
        ].copy()

        if len(df_dir) == 0:
            print(f"  ⚠️ No {sender}→{receiver} interactions")
            continue

        # Filter significant
        sig = df_dir[df_dir["magnitude_rank"] < 0.05].copy()
        if len(sig) == 0:
            print(f"  ⚠️ No significant {sender}→{receiver} interactions")
            continue

        # Deduplicate by (interaction, region)
        sig = (
            sig.groupby(["interaction", "region"], as_index=False)
            .agg({"lr_means": "mean", "magnitude_rank": "min"})
        )

        top_interactions = (
            sig.groupby("interaction")["lr_means"]
            .mean()
            .nlargest(20)
            .index.tolist()
        )
        plot_data = sig[sig["interaction"].isin(top_interactions)].copy()
        if len(plot_data) == 0:
            continue

        print(f"  Plotting {sender}→{receiver} ({len(plot_data)} points)")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.scatterplot(
            data=plot_data,
            x="region", y="interaction",
            size="lr_means", hue="lr_means",
            palette="viridis", sizes=(50, 400),
            edgecolor="none", alpha=0.75, ax=ax
        )

        ax.set_title(f"{sender} → {receiver}\nTop 20 Interactions Across Regions",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Region", fontsize=12)
        ax.set_ylabel("Ligand → Receptor", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = os.path.join(out_dir, f"bubble_{sender}_to_{receiver}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {out_file}")

print("\n✓ All directional bubble plots saved.")
