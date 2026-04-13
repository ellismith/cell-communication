#!/usr/bin/env python3
"""
Summary bubble plot of significant category enrichments across regions.
Strengthening and weakening shown as separate rows per category.

  x = region
  y = category × direction (strengthening above, weakening below)
  size = fold enrichment
  color = direction (red=strengthening, blue=weakening)
  opacity = % Louvain agreement
  annotation = top_sender → top_receiver
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── paths ──────────────────────────────────────────────────────────────────
SUMMARY_FILE = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/master_summary_table.csv"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions"

REGION_ORDER = ["ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCB", "M1", "MB", "mdTN", "NAc"]

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(SUMMARY_FILE)
print(f"Loaded {len(df)} rows")

categories = sorted(df["category"].unique())
region_pos = {r: i for i, r in enumerate(REGION_ORDER)}

# build y-axis: each category gets two rows — strengthening (top) and weakening (bottom)
# with a small gap between category groups
y_labels = []
y_pos_map = {}  # (category, direction) -> y position
y = 0
for cat in categories:
    y_pos_map[(cat, "strengthening")] = y
    y_labels.append(f"{cat}\n(str.)")
    y += 1
    y_pos_map[(cat, "weakening")] = y
    y_labels.append(f"{cat}\n(wk.)")
    y += 1.5  # gap between categories

max_fe     = df["fold_enrichment"].max()
size_scale = 500 / max_fe

n_rows = len(y_labels)
fig_w  = max(14, len(REGION_ORDER) * 1.1)
fig_h  = max(8,  n_rows * 0.55)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

for _, row in df.iterrows():
    if row["region"] not in region_pos:
        continue
    key = (row["category"], row["direction"])
    if key not in y_pos_map:
        continue

    xi    = region_pos[row["region"]]
    yi    = y_pos_map[key]
    size  = max(row["fold_enrichment"] * size_scale, 30)
    color = "#d62728" if row["direction"] == "strengthening" else "#1f77b4"
    alpha = min(0.4 + row["pct_louvain_agree"] / 100 * 0.6, 1.0)

    ax.scatter(xi, yi, s=size, c=color, alpha=alpha,
               edgecolors="white", linewidths=0.5, zorder=3)

    label = f"{row['top_sender']}→{row['top_receiver']}"
    ax.text(xi, yi, label,
            ha="center", va="center", fontsize=5.5,
            color="black", fontweight="bold", zorder=4)

# horizontal grid lines between category groups
for cat in categories:
    y_str = y_pos_map[(cat, "strengthening")]
    y_wk  = y_pos_map[(cat, "weakening")]
    ax.axhspan(y_str - 0.5, y_wk + 0.5, alpha=0.04, color="grey")

for j in range(len(REGION_ORDER)):
    ax.axvline(j, color="lightgrey", linewidth=0.5, zorder=1)

# y-axis ticks
all_y_pos = [y_pos_map[k] for k in sorted(y_pos_map.keys())]
ax.set_yticks(all_y_pos)
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_xticks(range(len(REGION_ORDER)))
ax.set_xticklabels(REGION_ORDER, fontsize=10)
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("Category × Direction", fontsize=11)
ax.set_title("Age-related signaling changes by category and region\n"
             "(size = fold enrichment, opacity = % Louvain agreement, label = top sender→receiver)",
             fontsize=11)
ax.set_xlim(-0.5, len(REGION_ORDER) - 0.5)
ax.set_ylim(-0.7, max(all_y_pos) + 0.7)

# legend
patches = [
    mpatches.Patch(color="#d62728", label="Strengthening"),
    mpatches.Patch(color="#1f77b4", label="Weakening"),
]
size_handles = [
    plt.scatter([], [], s=fe * size_scale, c="grey", alpha=0.7,
                label=f"FE={fe:.1f}")
    for fe in [1, 5, round(max_fe, 1)]
]
ax.legend(handles=patches + size_handles, loc="lower right",
          fontsize=8, framealpha=0.9)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "category_summary_bubble.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
