#!/usr/bin/env python3
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', default='HIP')
args = parser.parse_args()

region = args.region
qval_thresh = 0.05
base_dir = '/scratch/easmit31/cell_cell/results/within_region_analysis_corrected'
output_dir = os.path.join(base_dir, 'chord_plots_age_effects')
os.makedirs(output_dir, exist_ok=True)

reg_file = os.path.join(base_dir, f'regression_{region}', f'whole_{region.lower()}_age_sex_regression.csv')
df = pd.read_csv(reg_file)

df[['sender','receiver','ligand','receptor']] = df['interaction'].str.split('|', expand=True)
df['sender_broad'] = df['sender'].str.replace(r'_\d+$', '', regex=True)
df['receiver_broad'] = df['receiver'].str.replace(r'_\d+$', '', regex=True)

pos_df = df[(df['age_coef'] > 0) & (df['age_qval'] <= qval_thresh)]
neg_df = df[(df['age_coef'] < 0) & (df['age_qval'] <= qval_thresh)]

def make_chord_plot(df_subset, title, fname):
    if df_subset.empty:
        print(f"No significant edges for {title}, skipping.")
        return

    links = (df_subset.groupby(['sender_broad','receiver_broad'])['age_coef']
             .agg(weight=lambda x: x.abs().mean())
             .reset_index())
    links.columns = ['source','target','value']

    # get all unique nodes
    nodes = sorted(set(links['source']) | set(links['target']))
    node_ds = hv.Dataset(pd.DataFrame({'name': nodes}), 'name')

    chord = hv.Chord((links, node_ds))
    chord.opts(
        opts.Chord(
            title=title,
            edge_color='source',
            node_color='name',
            labels='name',
            cmap='Category10',
            edge_cmap='Category10',
            width=800, height=800
        )
    )
    hv.save(chord, os.path.join(output_dir, fname), fmt='png')
    print(f"Saved {fname}")

make_chord_plot(pos_df, f'{region} Age-related Increases', f'{region}_age_increases.png')
make_chord_plot(neg_df, f'{region} Age-related Decreases', f'{region}_age_decreases.png')
