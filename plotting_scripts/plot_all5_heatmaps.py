#!/usr/bin/env python3
"""
plot_all5_heatmaps.py
=====================
Plots heatmaps of age effect (OLS β) for 5 focal LR pairs:
  - CX3CL1|CX3CR1  (immune/microglial)
  - IL34|CSF1R      (immune/microglial)
  - NLGN1|NRXN1/2/3 (synaptic adhesion, one plot each)

Rows = ct_pairs with at least one significant Louvain at q_thresh,
       ranked by total n_sig louvains (desc), max top_n rows.
Cols = all 11 brain regions.
Numbers on cells = count of Louvain interactions with age_qval < q_thresh.

Usage
-----
python3 plot_all5_heatmaps.py
python3 plot_all5_heatmaps.py --top_n 20 --q_thresh 0.05
python3 plot_all5_heatmaps.py --q_thresh 0.1 --shared_cbar

Arguments
---------
--top_n       Max rows to show (default: 15)
--q_thresh    Q threshold for significance: selects rows, numbers cells,
              and ranks by n_sig (default: 0.05)
--shared_cbar If set, all 5 plots share the same colorbar scale

Output
------
results/manuscript_relevant_plots/heatmap_{pair}_top{N}_q{thresh}.png
"""
import pandas as pd, numpy as np, glob, os, argparse, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--top_n",       type=int,   default=15)
parser.add_argument("--q_thresh",    type=float, default=0.05)
parser.add_argument("--shared_cbar", action="store_true")
args = parser.parse_args()

DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_results"
OUT = "/scratch/easmit31/cell_cell/results/manuscript_relevant_plots"

ABBREV = {
    "Astrocyte":    "AST", "Microglia":    "MGL", "Oligo":        "OLIG",
    "OPC":          "OPC", "Ependymal":    "EPEN", "Vascular":     "VASC",
    "Glutamatergic":"EXC", "GABA":         "INH", "MSN":          "MSN",
    "Cerebellar":   "CER", "Midbrain":     "MBN", "Basket":       "BC",
}
REGION_LABELS = {
    "ACC":"ACC","CN":"CN","DLPFC":"dlPFC","EC":"EC","HIP":"HIP",
    "IPP":"IPP","LCB":"lCb","M1":"M1","MB":"MB","MDTN":"mdTN","NAC":"NAc"
}
ALL_REGIONS = ["ACC","CN","DLPFC","EC","HIP","IPP","LCB","M1","MB","MDTN","NAC"]
IMMUNE_PAIRS = {"CX3CL1|CX3CR1": ALL_REGIONS, "IL34|CSF1R": ALL_REGIONS}
NLGN_PAIRS   = ["NLGN1|NRXN1","NLGN1|NRXN2","NLGN1|NRXN3"]
ALL_PAIRS    = list(IMMUNE_PAIRS.keys()) + NLGN_PAIRS
CELL_SIZE    = 0.7

# ── load ──────────────────────────────────────────────────────────────────────
rows = []
for f in glob.glob(f"{DIR}/regression_*/whole_*_age_sex_regression.csv"):
    region = os.path.basename(f).replace("whole_","").split("_age_sex")[0].upper()
    df = pd.read_csv(f)
    p = df["interaction"].str.split("|",expand=True)
    df["sender_type"]   = p[0].str.replace(r"_\d+$","",regex=True)
    df["receiver_type"] = p[1].str.replace(r"_\d+$","",regex=True)
    df["lr_pair"] = p[2]+"|"+p[3]
    df["region"]  = region
    rows.append(df)
d = pd.concat(rows)
d = d[d["lr_pair"].isin(ALL_PAIRS)].copy()
d["sender_type"]   = d["sender_type"].map(ABBREV).fillna(d["sender_type"])
d["receiver_type"] = d["receiver_type"].map(ABBREV).fillna(d["receiver_type"])
d["ct_pair"] = d["sender_type"]+"→"+d["receiver_type"]

def get_mc_nsig(sub, regions):
    grp  = sub.groupby(["ct_pair","region"])
    mc   = grp["age_coef"].mean().unstack("region").reindex(columns=regions)
    nsig = grp["age_qval"].apply(lambda x: (x < args.q_thresh).sum()).unstack("region").reindex(columns=regions)
    nsig_label = grp["age_qval"].apply(lambda x: (x < 0.05).sum()).unstack("region").reindex(columns=regions)
    return mc, nsig, nsig_label

def rank_ct_pairs(sub, top_n):
    nsig_total = sub.groupby("ct_pair")["age_qval"].apply(lambda x: (x < args.q_thresh).sum())
    nsig_total = nsig_total[nsig_total > 0]
    return nsig_total.nlargest(top_n).index.tolist()

def make_plot(mc, nsig, nsig_label, vmax, pair, n_rows):
    n_cols = len(ALL_REGIONS)
    fig, ax = plt.subplots(figsize=(n_cols*CELL_SIZE+3, n_rows*CELL_SIZE+1))
    cols     = [c for c in ALL_REGIONS if c in mc.columns]
    col_labs = [REGION_LABELS[c] for c in cols]
    mc_p     = mc.reindex(columns=cols).rename(columns=REGION_LABELS)
    nsig_p       = nsig.reindex(columns=cols).rename(columns=REGION_LABELS)
    nsig_label_p = nsig_label.reindex(columns=cols).rename(columns=REGION_LABELS)

    sns.heatmap(mc_p, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor="gray", cbar=True,
                cbar_kws={"label":"Mean age effect (β)"})
    ax.collections[0].colorbar.ax.tick_params(labelsize=15)
    ax.collections[0].colorbar.set_label("Mean age effect (β)", fontsize=16)

    for i, row in enumerate(mc_p.index):
        for j, col in enumerate(col_labs):
            n = nsig_label_p.loc[row, col] if (row in nsig_label_p.index and col in nsig_label_p.columns) else np.nan
            if pd.notna(n) and n > 0:
                ax.text(j+0.5, i+0.5, str(int(n)), ha="center", va="center",
                        fontsize=13, color="black", fontweight="bold")

    ax.tick_params(axis="x", labelsize=16, rotation=45)
    ytick_size = 16
    ax.tick_params(axis="y", labelsize=ytick_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(pair, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Region", fontsize=17)
    ax.set_ylabel("Cell type pair", fontsize=17)
    plt.tight_layout()
    fname = os.path.join(OUT, f"heatmap_{pair.replace('|','_')}_top{args.top_n}_q{args.q_thresh}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ── global vmax ───────────────────────────────────────────────────────────────
global_vmax = None
if args.shared_cbar:
    all_vals = []
    for pair, regions in IMMUNE_PAIRS.items():
        sub = d[(d["lr_pair"]==pair) & (d["region"].isin(regions))]
        mc, _, __ = get_mc_nsig(sub, regions)
        all_vals.append(mc.values.flatten())
    for pair in NLGN_PAIRS:
        sub = d[(d["lr_pair"]==pair) & (d["region"].isin(ALL_REGIONS))]
        mc, _, __ = get_mc_nsig(sub, ALL_REGIONS)
        all_vals.append(mc.values.flatten())
    global_vmax = np.nanmax(np.abs(np.concatenate(all_vals)))
    print(f"Global vmax: {global_vmax:.4f}")

# ── immune ────────────────────────────────────────────────────────────────────
for pair, regions in IMMUNE_PAIRS.items():
    sub      = d[(d["lr_pair"]==pair) & (d["region"].isin(regions))].copy()
    mc, nsig, nsig_label = get_mc_nsig(sub, regions)
    ranked   = rank_ct_pairs(sub, args.top_n)
    if not ranked:
        print(f"No sig ct_pairs for {pair}"); continue
    mc   = mc.reindex(ranked)
    nsig = nsig.reindex(ranked)
    vals = mc.values[~np.isnan(mc.values)]
    vmax = global_vmax if args.shared_cbar else (np.nanmax(np.abs(vals)) if len(vals) else 0.01)
    make_plot(mc, nsig, nsig_label, vmax, pair, len(ranked))

# ── nlgn ──────────────────────────────────────────────────────────────────────
nlgn_sub    = d[d["lr_pair"].isin(NLGN_PAIRS)]
ranked_nlgn = rank_ct_pairs(nlgn_sub, args.top_n)
if not ranked_nlgn:
    print("No sig ct_pairs for NLGN pairs")
else:
    for pair in NLGN_PAIRS:
        sub = d[(d["lr_pair"]==pair) & (d["region"].isin(ALL_REGIONS))]
        mc, nsig, nsig_label = get_mc_nsig(sub, ALL_REGIONS)
        mc   = mc.reindex(ranked_nlgn)
        nsig = nsig.reindex(ranked_nlgn)
        vals = mc.values[~np.isnan(mc.values)]
        vmax = global_vmax if args.shared_cbar else (np.nanmax(np.abs(vals)) if len(vals) else 0.01)
        make_plot(mc, nsig, nsig_label, vmax, pair, len(ranked_nlgn))
