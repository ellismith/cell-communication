#!/usr/bin/env python3
"""
plot_directionality.py
-----------------------
Visualize directional ligand–receptor signaling between cell types
across all pairwise CellChat/Liana results.

Reads all *_results.csv files in /scratch/easmit31/cell_cell/results/pairwise/
and builds a chord diagram showing directional communication.

Usage:
    python plot_directionality.py
"""

import os, glob, pandas as pd, holoviews as hv
hv.extension('bokeh')

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Combine all CSVs
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["pair"] = os.path.basename(f).replace("_results.csv", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Filter significant interactions
sig = data[data["magnitude_rank"] < 0.05].copy()
if sig.empty:
    raise ValueError("No significant interactions found.")

# Aggregate by source → target
agg = sig.groupby(["source", "target"]).size().reset_index(name="count")

print(f"Total significant interactions: {len(sig):,}")
print(f"Unique source-target pairs: {len(agg):,}")

# Build chord
chord = hv.Chord(agg).opts(
    title="Cell–Cell Communication Directionality",
    cmap="viridis",
    edge_color="source",
    node_color="index",
    labels="index",
    width=700,
    height=700
)

outfile = os.path.join(out_dir, "chord_directionality.html")
hv.save(chord, outfile)
print(f"✓ Saved chord diagram to {outfile}")
