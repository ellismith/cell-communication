import pandas as pd

results = pd.read_csv("/scratch/easmit31/cell_cell/results/cellchat_significant_interactions.csv")

print("Column names:")
print(results.columns.tolist())

print("\nFirst few rows:")
print(results.head())

print("\n\nAstrocyte → GABA interactions:")
astro_to_gaba = results[(results['source'] == 'Astrocyte') & (results['target'] == 'GABA_neuron')]
print(astro_to_gaba)

print("\n\nGABA → Astrocyte interactions:")
gaba_to_astro = results[(results['source'] == 'GABA_neuron') & (results['target'] == 'Astrocyte')]
print(gaba_to_astro)
