#!/usr/bin/env python3
"""
plot_louvain_heatmaps.py
========================
Louvain-level heatmaps of age effect (OLS β) for 5 focal LR pairs.
Rows = individual sender_louvain→receiver_louvain combos,
       top N by n_sig louvains at q_thresh.
Cols = regions.
Color = raw age_coef (all cells with data shown).
Asterisk = significant at q_thresh.

Usage
-----
python3 plot_louvain_heatmaps.py
python3 plot_louvain_heatmaps.py --top_n 20 --q_thresh 0.05
python3 plot_louvain_heatmaps.py --align_cbar   # shared cbar: immune pairs share one, nlgn pairs share one
"""
import pandas as pd, numpy as np, glob, os, argparse, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--top_n",      type=int,   default=15)
parser.add_argument("--q_thresh",   type=float, default=0.05)
parser.add_argument("--align_cbar", action="store_true",
                    help="share colorbar: immune pairs share scale, nlgn pairs share scale")
args = parser.parse_args()

DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_results"
OUT = "/scratch/easmit31/cell_cell/results/manuscript_relevant_plots"

PAIRS        = ["NLGN1|NRXN1","NLGN1|NRXN2","NLGN1|NRXN3","CX3CL1|CX3CR1","IL34|CSF1R"]
IMMUNE_PAIRS = {"CX3CL1|CX3CR1","IL34|CSF1R"}
NLGN_PAIRS   = {"NLGN1|NRXN1","NLGN1|NRXN2","NLGN1|NRXN3"}
REGIONS      = ["ACC","CN","DLPFC","EC","HIP","IPP","LCB","M1","MB","MDTN","NAC"]
REGION_LABELS = {
    "ACC":"ACC","CN":"CN","DLPFC":"dlPFC","EC":"EC","HIP":"HIP",
    "IPP":"IPP","LCB":"lCb","M1":"M1","MB":"MB","MDTN":"mdTN","NAC":"NAc"
}
ABBREV = {
    "Astrocyte":    "AST", "Microglia":    "MGL", "Oligo":        "OLIG",
    "OPC":          "OPC", "Ependymal":    "EPEN", "Vascular":     "VASC",
    "Glutamatergic":"EXC", "GABA":         "INH", "MSN":          "MSN",
    "Cerebellar":   "CER", "Midbrain":     "MBN", "Basket":       "BC",
}
CELL_WIDTH  = 2.2
CELL_HEIGHT = 1.05

def abbrev_louvain(s):
    for k, v in ABBREV.items():
        if s.startswith(k):
            rest = s[len(k):].lstrip('_')
            return f"{v}_{rest}" if rest else v
    return s

def nat_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# ── load ──────────────────────────────────────────────────────────────────────
rows = []
for f in glob.glob(f"{DIR}/regression_*/whole_*_age_sex_regression.csv"):
    region = os.path.basename(f).replace("whole_","").split("_age_sex")[0].upper()
    df = pd.read_csv(f)
    p = df["interaction"].str.split("|",expand=True)
    df["sender_louvain"]   = p[0]
    df["receiver_louvain"] = p[1]
    df["lr_pair"] = p[2]+"|"+p[3]
    df["region"]  = region
    rows.append(df)
d = pd.concat(rows)
d = d[d["lr_pair"].isin(PAIRS)].copy()
d["louvain_pair"] = d["sender_louvain"].apply(abbrev_louvain)+"→"+d["receiver_louvain"].apply(abbrev_louvain)

# ── compute group vmaxes if align_cbar ───────────────────────────────────────
immune_vmax = None
nlgn_vmax   = None
if args.align_cbar:
    for group, pairs in [("immune", IMMUNE_PAIRS), ("nlgn", NLGN_PAIRS)]:
        vals = []
        for pair in pairs:
            sub = d[d["lr_pair"]==pair]
            sig = sub[sub["age_qval"] < args.q_thresh]
            if sig.empty: continue
            top_pairs = sig.groupby("louvain_pair")["age_qval"].count().nlargest(args.top_n).index.tolist()
            sub_top = sub[sub["louvain_pair"].isin(top_pairs)]
            piv = sub_top.pivot_table(index="louvain_pair", columns="region",
                                      values="age_coef", aggfunc="mean").reindex(columns=REGIONS)
            vals.append(piv.values.flatten())
        if vals:
            vmax = np.nanmax(np.abs(np.concatenate(vals)))
            if group == "immune": immune_vmax = vmax
            else: nlgn_vmax = vmax
    print(f"Immune vmax: {immune_vmax:.4f}, NLGN vmax: {nlgn_vmax:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
for pair in PAIRS:
    sub = d[d["lr_pair"]==pair].copy()
    sig = sub[sub["age_qval"] < args.q_thresh]
    if sig.empty:
        print(f"No sig interactions for {pair}"); continue

    if pair in IMMUNE_PAIRS:
        top_pairs = sorted(sig["louvain_pair"].unique().tolist(), key=nat_key)
    else:
        top_pairs = (sig.groupby("louvain_pair")["age_qval"]
                     .count().nlargest(args.top_n).index.tolist())
        top_pairs = sorted(top_pairs, key=nat_key)

    sub_top  = sub[sub["louvain_pair"].isin(top_pairs)].copy()
    piv      = sub_top.pivot_table(index="louvain_pair", columns="region",
                                   values="age_coef", aggfunc="mean").reindex(columns=REGIONS)
    sig_mask = sub_top.pivot_table(index="louvain_pair", columns="region",
                                   values="age_qval",
                                   aggfunc=lambda x: (x < args.q_thresh).any()).reindex(columns=REGIONS)
    piv      = piv.reindex(top_pairs)
    sig_mask = sig_mask.reindex(top_pairs)

    col_labs = [REGION_LABELS.get(c, c) for c in REGIONS]
    piv.columns      = col_labs
    sig_mask.columns = col_labs

    if args.align_cbar:
        vmax = immune_vmax if pair in IMMUNE_PAIRS else nlgn_vmax
    else:
        vmax = np.nanmax(np.abs(piv.values))

    n_rows = len(top_pairs)
    n_cols = len(REGIONS)
    fig, ax = plt.subplots(figsize=(n_cols*CELL_WIDTH+5, n_rows*CELL_HEIGHT+3))

    sns.heatmap(piv, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.3, linecolor="gray", cbar_kws={"label":"Age effect (β)"})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=34)
    cbar.set_label("Age effect (β)", fontsize=44)

    for i, row in enumerate(piv.index):
        for j, col in enumerate(col_labs):
            if sig_mask.loc[row, col] == True:
                ax.text(j+0.5, i+0.5, "*", ha="center", va="center",
                        fontsize=30, color="white", fontweight="bold")

    ax.set_title(pair, fontsize=40, fontweight="bold", pad=16)
    ax.set_xlabel("Region", fontsize=44)
    ax.set_ylabel("Sender→Receiver (Louvain)", fontsize=44)
    ax.tick_params(axis="x", labelsize=34, rotation=45)
    ax.tick_params(axis="y", labelsize=30, rotation=0)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)

    plt.tight_layout()
    suffix = "_aligncbar" if args.align_cbar else ""
    fname = os.path.join(OUT, f"heatmap_louvain_{pair.replace('|','_')}_q{args.q_thresh}{suffix}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
