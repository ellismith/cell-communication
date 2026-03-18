#!/usr/bin/env python3
"""
Build LR functional annotations from CellPhoneDB classification column.
Maps our liana-format LR pairs to CellPhoneDB classifications via ligand matching.
Unmatched ligands are annotated manually with _Claude suffix.

Usage:
    python build_cpdb_annotations.py
"""

import pandas as pd
from pathlib import Path

# ── LOAD ──────────────────────────────────────────────────────────────────────

cpdb = pd.read_csv('/scratch/easmit31/cell_cell/CellPhoneDB_latest/interaction_input_CellPhoneDB.csv')
gene_input = pd.read_csv('/scratch/easmit31/cell_cell/CellPhoneDB-data/data/gene_input.csv')
liana = pd.read_csv('/scratch/easmit31/cell_cell/cellphonedb_interactions_liana_format.csv')

# ── MAP PARTNER ACCESSIONS TO GENE SYMBOLS ────────────────────────────────────

prot2gene = dict(zip(gene_input['uniprot'], gene_input['gene_name']))
cpdb['gene_a'] = cpdb['partner_a'].map(prot2gene)

ligand_class = (cpdb.dropna(subset=['gene_a', 'classification'])
                .groupby('gene_a')['classification']
                .agg(lambda x: x.mode()[0]))

liana['lr_pair'] = liana['ligand'] + '|' + liana['receptor']
liana['classification'] = liana['ligand'].map(ligand_class)

print(f"Total liana pairs: {len(liana)}")
print(f"Matched by ligand: {liana['classification'].notna().sum()}")
print(f"Unmatched: {liana['classification'].isna().sum()}")

# ── MANUAL ANNOTATIONS FOR UNMATCHED LIGANDS (_Claude suffix) ─────────────────

MANUAL_LIGAND_CLASS = {
    'GLS': 'Signaling by Glutamate_Claude', 'GLS2': 'Signaling by Glutamate_Claude',
    'SLC1A1': 'Signaling by Glutamate_Claude', 'SLC1A2': 'Signaling by Glutamate_Claude',
    'SLC1A3': 'Signaling by Glutamate_Claude', 'SLC1A6': 'Signaling by Glutamate_Claude',
    'SLC1A7': 'Signaling by Glutamate_Claude', 'SLC17A6': 'Signaling by Glutamate_Claude',
    'SLC17A7': 'Signaling by Glutamate_Claude', 'SLC17A8': 'Signaling by Glutamate_Claude',
    'SHMT1': 'Signaling by Glutamate_Claude', 'SHMT2': 'Signaling by Glutamate_Claude',
    'GAD1': 'Signaling by GABA_Claude', 'GAD2': 'Signaling by GABA_Claude',
    'SLC6A1': 'Signaling by GABA_Claude', 'SLC6A11': 'Signaling by GABA_Claude',
    'SLC6A12': 'Signaling by GABA_Claude', 'SLC6A13': 'Signaling by GABA_Claude',
    'SLC32A1': 'Signaling by GABA_Claude',
    'DDC': 'Signaling by Serotonin/Dopamine_Claude', 'DBH': 'Signaling by Serotonin/Dopamine_Claude',
    'PNMT': 'Signaling by Serotonin/Dopamine_Claude', 'SLC6A2': 'Signaling by Serotonin/Dopamine_Claude',
    'SLC6A3': 'Signaling by Serotonin/Dopamine_Claude', 'SLC18A1': 'Signaling by Serotonin/Dopamine_Claude',
    'SLC18A2': 'Signaling by Serotonin/Dopamine_Claude', 'SLC6A6': 'Signaling by Serotonin/Dopamine_Claude',
    'SLC6A8': 'Signaling by Serotonin/Dopamine_Claude', 'SLC10A4': 'Signaling by Serotonin/Dopamine_Claude',
    'TPH1': 'Signaling by Serotonin_Claude', 'TPH2': 'Signaling by Serotonin_Claude',
    'SLC6A4': 'Signaling by Serotonin_Claude', 'ASMT': 'Signaling by Serotonin_Claude',
    'HDC': 'Signaling by Histamine_Claude',
    'CHAT': 'Signaling by Acetylcholine_Claude', 'SLC18A3': 'Signaling by Acetylcholine_Claude',
    'SLC5A7': 'Signaling by Acetylcholine_Claude',
    'SLC6A5': 'Signaling by Glycine_Claude', 'SLC6A9': 'Signaling by Glycine_Claude',
    'SLC29A1': 'Signaling by Adenosine_Claude', 'SLC29A2': 'Signaling by Adenosine_Claude',
    'SLC29A3': 'Signaling by Adenosine_Claude', 'SLC29A4': 'Signaling by Adenosine_Claude',
    'NT5E': 'Signaling by Adenosine_Claude',
    'CYP11A1': 'Signaling by Steroids_Claude', 'CYP11B1': 'Signaling by Steroids_Claude',
    'CYP11B2': 'Signaling by Steroids_Claude', 'CYP17A1': 'Signaling by Steroids_Claude',
    'CYP19A1': 'Signaling by Steroids_Claude', 'CYP21A2': 'Signaling by Steroids_Claude',
    'CYP27B1': 'Signaling by Steroids_Claude', 'CYP3A4': 'Signaling by Steroids_Claude',
    'HSD17B1': 'Signaling by Steroids_Claude', 'HSD17B12': 'Signaling by Steroids_Claude',
    'HSD17B3': 'Signaling by Steroids_Claude', 'HSD17B6': 'Signaling by Steroids_Claude',
    'HSD3B1': 'Signaling by Steroids_Claude', 'SRD5A1': 'Signaling by Steroids_Claude',
    'SRD5A2': 'Signaling by Steroids_Claude', 'SRD5A3': 'Signaling by Steroids_Claude',
    'SULT1A1': 'Signaling by Steroids_Claude', 'SULT1E1': 'Signaling by Steroids_Claude',
    'SULT2A1': 'Signaling by Steroids_Claude', 'AKR1B1': 'Signaling by Steroids_Claude',
    'AKR1C3': 'Signaling by Steroids_Claude', 'AKR1D1': 'Signaling by Steroids_Claude',
    'DHRS9': 'Signaling by Steroids_Claude', 'FDX1': 'Signaling by Steroids_Claude',
    'FDX2': 'Signaling by Steroids_Claude', 'FDXR': 'Signaling by Steroids_Claude',
    'TG': 'Signaling by Steroids_Claude', 'TPO': 'Signaling by Steroids_Claude',
    'TYR': 'Signaling by Steroids_Claude',
    'DHCR24': 'Signaling by Cholesterol_Claude', 'DHCR7': 'Signaling by Cholesterol_Claude',
    'ALDH1A1': 'Signaling by Retinoic Acid_Claude', 'ALDH1A2': 'Signaling by Retinoic Acid_Claude',
    'ALDH1A3': 'Signaling by Retinoic Acid_Claude', 'DIO3': 'Signaling by Retinoic Acid_Claude',
    'RBP4': 'Signaling by Retinoic Acid_Claude',
    'ALOX12': 'Signaling by Lipoxin/Leukotriene_Claude', 'ALOX5': 'Signaling by Lipoxin/Leukotriene_Claude',
    'ALOX5AP': 'Signaling by Lipoxin/Leukotriene_Claude', 'LTA4H': 'Signaling by Lipoxin/Leukotriene_Claude',
    'LTC4S': 'Signaling by Lipoxin/Leukotriene_Claude', 'DPEP1': 'Signaling by Lipoxin/Leukotriene_Claude',
    'DPEP2': 'Signaling by Lipoxin/Leukotriene_Claude', 'DPEP3': 'Signaling by Lipoxin/Leukotriene_Claude',
    'GGT1': 'Signaling by Lipoxin/Leukotriene_Claude', 'GGT5': 'Signaling by Lipoxin/Leukotriene_Claude',
    'PTGDS': 'Signaling by Prostaglandin_Claude', 'PTGES': 'Signaling by Prostaglandin_Claude',
    'PTGES2': 'Signaling by Prostaglandin_Claude', 'PTGES3': 'Signaling by Prostaglandin_Claude',
    'PTGIS': 'Signaling by Prostaglandin_Claude', 'PTGR1': 'Signaling by Prostaglandin_Claude',
    'TBXAS1': 'Signaling by Thromboxane_Claude', 'CBR1': 'Signaling by Prostaglandin_Claude',
    'HEBP1': 'Signaling by Prostaglandin_Claude',
    'DAGLA': 'Signaling by Arachidonoylglycerol_Claude', 'DAGLB': 'Signaling by Arachidonoylglycerol_Claude',
    'GDNF': 'Signaling by Neurotrophin_Claude', 'NELL2': 'Signaling by Neurotrophin_Claude',
    'CBLN1': 'Signaling by Neurotrophin_Claude', 'NXNL1': 'Signaling by Neurotrophin_Claude',
    'TULP1': 'Signaling by Neurotrophin_Claude', 'GAS6': 'Signaling by Neurotrophin_Claude',
    'DAG1': 'Signaling by Neurexin_Claude', 'UNC5A': 'Signaling by Netrin_Claude',
    'LRFN4': 'Signaling by Leucine Rich Repeat And Fibronectin_Claude',
    'LRFN5': 'Signaling by Leucine Rich Repeat And Fibronectin_Claude',
    'LRRC4B': 'Signaling by Leucine-Rich Repeat Transmembrane Neuronal Protein_Claude',
    'LRRC4C': 'Signaling by Leucine-Rich Repeat Transmembrane Neuronal Protein_Claude',
    'NECTIN3': 'Signaling by Nectin_Claude',
    'ADCYAP1': 'Signaling by Pituitary adenylate_Claude',
    'ADM': 'Signaling by Adrenomedullin_Claude', 'ADM2': 'Signaling by Adrenomedullin_Claude',
    'APELA': 'Signaling by Apelin_Claude', 'APLN': 'Signaling by Apelin_Claude',
    'CALCA': 'Signaling by Calcitonin_Claude', 'CALCB': 'Signaling by Calcitonin_Claude',
    'CCK': 'Signaling by Cholecystokinin_Claude',
    'CRH': 'Signaling by Corticotropin-releasing factor_Claude',
    'GAST': 'Signaling by Gastrin_Claude', 'GIP': 'Signaling by Neuropeptide precursor_Claude',
    'GNRH1': 'Signaling by Progonadoliberin_Claude', 'GNRH2': 'Signaling by Progonadoliberin_Claude',
    'GRP': 'Signaling by Neuropeptide precursor_Claude',
    'IAPP': 'Signaling by Amylin_Claude', 'INS': 'Signaling by Insulin-like precursor_Claude',
    'INSL3': 'Signaling by Relaxin_Claude', 'INSL5': 'Signaling by Relaxin_Claude',
    'PMCH': 'Signaling by Pro-MHC_Claude', 'PRLH': 'Signaling by Prolactin_Claude',
    'PROK1': 'Signaling by Prokineticin_Claude', 'PROK2': 'Signaling by Prokineticin_Claude',
    'SCT': 'Signaling by Secretin_Claude', 'SPX': 'Signaling by Spexin precursor_Claude',
    'TAFA1': 'Signaling by TAFA_Claude', 'TAFA4': 'Signaling by TAFA_Claude',
    'TAFA5': 'Signaling by TAFA_Claude',
    'LAMA2': 'Adhesion by Laminin_Claude', 'FBN1': 'Adhesion by Fibronectin_Claude',
    'FGA': 'Adhesion by Fibribogen_Claude', 'FGB': 'Adhesion by Fibribogen_Claude',
    'FGG': 'Adhesion by Fibribogen_Claude', 'CSPG4': 'Adhesion by Collagen/Integrin_Claude',
    'OMD': 'Adhesion by Collagen/Integrin_Claude', 'ITGAV': 'Adhesion by Collagen/Integrin_Claude',
    'ITGB1': 'Adhesion by Collagen/Integrin_Claude', 'ITGB3': 'Adhesion by Collagen/Integrin_Claude',
    'ITGB5': 'Adhesion by Collagen/Integrin_Claude', 'LRPAP1': 'Adhesion by Collagen/Integrin_Claude',
    'TGM2': 'Adhesion by Fibronectin_Claude', 'ALCAM': 'Adhesion by CADM_Claude',
    'CHL1': 'Adhesion by L1CAM_Claude', 'MADCAM1': 'Adhesion by VCAM_Claude',
    'ESAM': 'Adhesion by JAM_Claude', 'PECAM1': 'Adhesion by ICAM_Claude',
    'THBS2': 'Adhesion by Thrombospondin_Claude',
    'CD274': 'Signaling by Tumor necrosis factor_Claude',
    'CD276': 'Signaling by Tumor necrosis factor_Claude',
    'PDCD1LG2': 'Signaling by Tumor necrosis factor_Claude',
    'CD70': 'Signaling by Tumor necrosis factor_Claude',
    'CD80': 'Signaling by Interleukin_Claude', 'CD86': 'Signaling by Interleukin_Claude',
    'CD40LG': 'Signaling by Interleukin_Claude', 'ICOSLG': 'Signaling by Interleukin_Claude',
    'IL12A': 'Signaling by Interleukin_Claude', 'IL12B': 'Signaling by Interleukin_Claude',
    'IL17A': 'Signaling by Interleukin_Claude', 'IL23A': 'Signaling by Interleukin_Claude',
    'IL27': 'Signaling by Interleukin_Claude', 'EBI3': 'Signaling by Interleukin_Claude',
    'CLCF1': 'Signaling by Interleukin_Claude', 'CNTF': 'Signaling by Interleukin_Claude',
    'LIF': 'Signaling by Interleukin_Claude', 'EPO': 'Signaling by Interleukin_Claude',
    'TSLP': 'Signaling by Interleukin_Claude', 'FLT3LG': 'Signaling by Interleukin_Claude',
    'THPO': 'Signaling by Interleukin_Claude', 'KITLG': 'Signaling by Interleukin_Claude',
    'LEP': 'Signaling by Interleukin_Claude', 'ADIPOQ': 'Signaling by Interleukin_Claude',
    'MST1': 'Signaling by Interleukin_Claude', 'TYROBP': 'Signaling by Interleukin_Claude',
    'PRXL2B': 'Signaling by Interleukin_Claude', 'BST2': 'Signaling by Interleukin_Claude',
    'CRLF2': 'Signaling by Interleukin_Claude', 'OSM': 'Signaling by Oncostatin-M precursor_Claude',
    'PRL': 'Signaling by Prolactin_Claude',
    'XCL1': 'Signaling by Chemokines_Claude', 'XCL2': 'Signaling by Chemokines_Claude',
    'CLEC2A': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'CLEC2B': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'CD93': 'Signaling by Complement_Claude', 'SFTPD': 'Signaling by Complement_Claude',
    'LAIR1': 'Signaling by FC receptor_Claude', 'PTPRC': 'Signaling by FC receptor_Claude',
    'LCK': 'Signaling by FC receptor_Claude', 'UBASH3B': 'Signaling by FC receptor_Claude',
    'KLRB1': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'CD96': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'MICA': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'MICB': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'NCR3LG1': 'Signaling by Killer Cell Lectin Like Receptor K1_Claude',
    'PVR': 'Signaling by Poliovirus receptor_Claude',
    'SIGLEC15': 'Signaling by Chemokines_Claude', 'CRTAM': 'Signaling by Chemokines_Claude',
    'CD44': 'Signaling by Chemokines_Claude', 'CD47': 'Signaling by Chemokines_Claude',
    'CD48': 'Signaling by Chemokines_Claude', 'CD52': 'Signaling by Chemokines_Claude',
    'CD55': 'Signaling by Chemokines_Claude', 'CD58': 'Signaling by Chemokines_Claude',
    'CD99': 'Signaling by Chemokines_Claude', 'CD200': 'Signaling by Chemokines_Claude',
    'CD226': 'Signaling by Chemokines_Claude', 'CD24': 'Signaling by Chemokines_Claude',
    'CD248': 'Signaling by Chemokines_Claude', 'CD34': 'Signaling by Chemokines_Claude',
    'SPN': 'Signaling by Chemokines_Claude', 'MPZL1': 'Signaling by Chemokines_Claude',
    'PPIA': 'Signaling by Chemokines_Claude', 'C10orf99': 'Signaling by Chemokines_Claude',
    'CD177': 'Signaling by Chemokines_Claude', 'RNASET2': 'Signaling by Chemokines_Claude',
    'PF4': 'Signaling by Chemokines_Claude',
    'B2M': 'Signaling by HLA_Claude', 'BAG6': 'Signaling by HLA_Claude',
    'CD1B': 'Signaling by HLA_Claude', 'CD1D': 'Signaling by HLA_Claude',
    'ERVH48-1': 'Signaling by HLA_Claude',
    'CD320': 'Signaling by Transferrin_Claude', 'ALB': 'Signaling by Transferrin_Claude',
    'LCN2': 'Signaling by Transferrin_Claude', 'FTH1': 'Signaling by Transferrin_Claude',
    'FTL': 'Signaling by Transferrin_Claude', 'HFE': 'Signaling by Transferrin_Claude',
    'CGA': 'Signaling by Choriogonadotropin_Claude', 'CGB1': 'Signaling by Choriogonadotropin_Claude',
    'CGB2': 'Signaling by Choriogonadotropin_Claude', 'CGB3': 'Signaling by Choriogonadotropin_Claude',
    'CGB7': 'Signaling by Choriogonadotropin_Claude', 'FSHB': 'Signaling by Choriogonadotropin_Claude',
    'LHB': 'Signaling by Choriogonadotropin_Claude', 'TSHB': 'Signaling by Choriogonadotropin_Claude',
    'GPHA2': 'Signaling by Choriogonadotropin_Claude', 'GPHB5': 'Signaling by Choriogonadotropin_Claude',
    'AMH': 'Signaling by Muellerian-inhibiting factor_Claude',
    'LEFTY1': 'Signaling by Left-right determination factor_Claude',
    'LEFTY2': 'Signaling by Left-right determination factor_Claude',
    'SOSTDC1': 'Signaling by Sclerostin domain-containing protein_Claude',
    'TRH': 'Signaling by Neuropeptide precursor_Claude', 'MLN': 'Signaling by Neuropeptide precursor_Claude',
    'QRFP': 'Signaling by Neuropeptide precursor_Claude', 'KISS1': 'Signaling by Neuropeptide precursor_Claude',
    'NSMF': 'Signaling by Neuropeptide precursor_Claude', 'REN': 'Signaling by Angiotensinogen_Claude',
    'KNG1': 'Signaling by Kininogen_Claude',
    'HGF': 'Signaling by Fibroblast growth factor_Claude',
    'NDP': 'Signaling by Vascular endothelial growth factor_Claude',
    'PGF': 'Signaling by Placenta growth factor_Claude',
    'FGFR4': 'Signaling by Fibroblast growth factor_Claude',
    'PI16': 'Signaling by Fibroblast growth factor_Claude',
    'BMPR1B': 'Signaling by BMP_Claude', 'BMPR2': 'Signaling by BMP_Claude',
    'INHBA': 'Signaling by Inhibin/Activin_Claude', 'INHBB': 'Signaling by Inhibin/Activin_Claude',
    'LIPA': 'Signaling by Apolipoprotein_Claude', 'CEL': 'Signaling by Apolipoprotein_Claude',
    'CLU': 'Signaling by Apolipoprotein_Claude', 'SAA1': 'Signaling by Apolipoprotein_Claude',
    'PRNP': 'Signaling by Amyloid-beta precursor protein_Claude',
    'RTN4R': 'Signaling by Amyloid-like protein_Claude',
    'PROC': 'Signaling by Plasminogen Activator_Claude', 'PLG': 'Signaling by Plasminogen Activator_Claude',
    'MMP2': 'Signaling by Plasminogen Activator_Claude',
    'PTPRD': 'Signaling by RET receptors_Claude', 'PTPRS': 'Signaling by RET receptors_Claude',
    'GP1BA': 'Signaling by von Willebrand factor_Claude',
    'MT-RNR2': 'Signaling by Humanin_Claude',
}

liana['classification'] = liana.apply(
    lambda r: MANUAL_LIGAND_CLASS.get(r['ligand'], r['classification'])
    if pd.isna(r['classification']) else r['classification'],
    axis=1
)

still_unmatched = liana[liana['classification'].isna()]['ligand'].unique()
print(f"Still unmatched after manual: {len(still_unmatched)}")
if len(still_unmatched) > 0:
    print(still_unmatched.tolist())

# ── DEFINE BROAD CATEGORY GROUPINGS ───────────────────────────────────────────

CATEGORY_MAP = {
    'Signaling by Neuroligin':                                  'Synaptic adhesion',
    'Signaling by Neurexin':                                    'Synaptic adhesion',
    'Signaling by Leucine-Rich Repeat Transmembrane Neuronal Protein': 'Synaptic adhesion',
    'Signaling by Calsyntenin':                                 'Synaptic adhesion',
    'Signaling by SLIT and NTRK-like protein':                  'Synaptic adhesion',
    'Signaling by Leucine Rich Repeat And Fibronectin':         'Synaptic adhesion',
    'Signaling by Teneurin':                                    'Teneurin/Latrophilin',
    'Signaling by Glutamate':                                   'Glutamate signaling',
    'Signaling by GABA':                                        'GABA signaling',
    'Signaling by Glycine':                                     'Glycine signaling',
    'Signaling by Acetylcholine':                               'Cholinergic signaling',
    'Signaling by Choline acetyltransferase':                   'Cholinergic signaling',
    'Signaling by Serotonin':                                   'Monoamine signaling',
    'Signaling by Serotonin/Dopamine':                          'Monoamine signaling',
    'Signaling by Noradrenaline':                               'Monoamine signaling',
    'Signaling by Adrenaline':                                  'Monoamine signaling',
    'Signaling by Epinephrine':                                 'Monoamine signaling',
    'Signaling by Histamine':                                   'Monoamine signaling',
    'Signaling by Neuropeptide precursor':                      'Neuropeptide signaling',
    'Signaling by Somatostatin':                                'Neuropeptide signaling',
    'Signaling by Galanin':                                     'Neuropeptide signaling',
    'Signaling by Vasoactive Intestinal Peptide':               'Neuropeptide signaling',
    'Signaling by Opioid':                                      'Neuropeptide signaling',
    'Signaling by Oxytocin':                                    'Neuropeptide signaling',
    'Signaling by Vasopressin':                                 'Neuropeptide signaling',
    'Signaling by Orexin':                                      'Neuropeptide signaling',
    'Signaling by Neurotensin':                                 'Neuropeptide signaling',
    'Signaling by Tachykinins':                                 'Neuropeptide signaling',
    'Signaling by Neuromedin':                                  'Neuropeptide signaling',
    'Signaling by Peptide YY':                                  'Neuropeptide signaling',
    'Signaling by Proopiomelanocortin':                         'Neuropeptide signaling',
    'Signaling by Agouti-related protein':                      'Neuropeptide signaling',
    'Signaling by Pro-glucagon precursor':                      'Neuropeptide signaling',
    'Signaling by Urocortin/Urotensin':                         'Neuropeptide signaling',
    'Signaling by Adenosine':                                   'Neuropeptide signaling',
    'Signaling by Adrenomedullin':                              'Neuropeptide signaling',
    'Signaling by Calcitonin':                                  'Neuropeptide signaling',
    'Signaling by Cholecystokinin':                             'Neuropeptide signaling',
    'Signaling by Corticotropin-releasing factor':              'Neuropeptide signaling',
    'Signaling by Gastrin':                                     'Neuropeptide signaling',
    'Signaling by Amylin':                                      'Neuropeptide signaling',
    'Signaling by Prokineticin':                                'Neuropeptide signaling',
    'Signaling by Secretin':                                    'Neuropeptide signaling',
    'Signaling by TAFA':                                        'Neuropeptide signaling',
    'Signaling by Pituitary adenylate':                         'Neuropeptide signaling',
    'Signaling by Apelin':                                      'Neuropeptide signaling',
    'Signaling by Spexin precursor':                            'Neuropeptide signaling',
    'Signaling by Pro-MHC':                                     'Neuropeptide signaling',
    'Signaling by Ephrin':                                      'Axon guidance',
    'Signaling by Semaphorin':                                  'Axon guidance',
    'Signaling by Netrin':                                      'Axon guidance',
    'Signaling by Neurotrophin':                                'Neurotrophic',
    'Signaling by Brain-derived neurotrophic factor':           'Neurotrophic',
    'Signaling by Beta-nerve growth factor':                    'Neurotrophic',
    'Signaling by Neuregulin':                                  'Neurotrophic',
    'Signaling by Pro-neuregulin':                              'Neurotrophic',
    'Signaling by Fibroblast growth factor':                    'Growth factor signaling',
    'Signaling by Epidermal growth factor':                     'Growth factor signaling',
    'Signaling by Insulin-like growth factor':                  'Growth factor signaling',
    'Signaling by Platelet-derived growth factor':              'Growth factor signaling',
    'Signaling by Vascular endothelial growth factor':          'Growth factor signaling',
    'Signaling by Transforming growth factor':                  'Growth factor signaling',
    'Signaling by BMP':                                         'Growth factor signaling',
    'Signaling by Growth differentiation factor':               'Growth factor signaling',
    'Signaling by Inhibin/Activin':                             'Growth factor signaling',
    'Signaling by NODAL':                                       'Growth factor signaling',
    'Signaling by Reelin':                                      'Growth factor signaling',
    'Signaling by Agrin':                                       'Growth factor signaling',
    'Signaling by Muellerian-inhibiting factor':                'Growth factor signaling',
    'Signaling by Betacellulin':                                'Growth factor signaling',
    'Signaling by Placenta growth factor':                      'Growth factor signaling',
    'Signaling by Left-right determination factor':             'Growth factor signaling',
    'Signaling by Sclerostin domain-containing protein':        'Wnt signaling',
    'Signaling by Chemokines':                                  'Immune/Inflammatory',
    'Signaling by Interleukin':                                 'Immune/Inflammatory',
    'Signaling by Tumor necrosis factor':                       'Immune/Inflammatory',
    'Signaling by Interferon':                                  'Immune/Inflammatory',
    'Signaling by Complement':                                  'Immune/Inflammatory',
    'Signaling by HLA':                                         'Immune/Inflammatory',
    'Signaling by Colony-Stimulating factor':                   'Immune/Inflammatory',
    'Signaling by Killer Cell Lectin Like Receptor K1':         'Immune/Inflammatory',
    'Signaling by FC receptor':                                 'Immune/Inflammatory',
    'Signaling by Leptin':                                      'Immune/Inflammatory',
    'Signaling by Lymphotactin':                                'Immune/Inflammatory',
    'Signaling by Oncostatin-M precursor':                      'Immune/Inflammatory',
    'Signaling by Poliovirus receptor':                         'Immune/Inflammatory',
    'Adhesion by Collagen/Integrin':                            'ECM/Adhesion',
    'Adhesion by Fibronectin':                                  'ECM/Adhesion',
    'Adhesion by Laminin':                                      'ECM/Adhesion',
    'Adhesion by ICAM':                                         'ECM/Adhesion',
    'Adhesion by Cadherin':                                     'ECM/Adhesion',
    'Adhesion by CADM':                                         'ECM/Adhesion',
    'Adhesion by L1CAM':                                        'ECM/Adhesion',
    'Adhesion by tenascin':                                     'ECM/Adhesion',
    'Adhesion by Vitronectin':                                  'ECM/Adhesion',
    'Adhesion by Thrombospondin':                               'ECM/Adhesion',
    'Adhesion by Osteopontin':                                  'ECM/Adhesion',
    'Adhesion by Desmosome':                                    'ECM/Adhesion',
    'Adhesion by Nectin':                                       'ECM/Adhesion',
    'Adhesion by VCAM':                                         'ECM/Adhesion',
    'Adhesion by JAM':                                          'ECM/Adhesion',
    'Adhesion by CEAM':                                         'ECM/Adhesion',
    'Adhesion by antigen THY1':                                 'ECM/Adhesion',
    'Signaling by Integrin':                                    'ECM/Adhesion',
    'Signaling by Fibronectin':                                 'ECM/Adhesion',
    'Signaling by Podocalyxin-like protein':                    'ECM/Adhesion',
    'Signaling by Nectin':                                      'ECM/Adhesion',
    'Signaling by Thrombospondin precursor':                    'ECM/Adhesion',
    'Adhesion by GAP junction':                                 'Gap junction',
    'Signaling by WNT':                                         'Wnt signaling',
    'Signaling by WNT inhibition':                              'Wnt signaling',
    'Signaling by R-spondin':                                   'Wnt signaling',
    'Signaling by Notch':                                       'Notch/Hedgehog',
    'Signaling by Sonic hedgehog':                              'Notch/Hedgehog',
    'Signaling by Desert hedgehog':                             'Notch/Hedgehog',
    'Signaling by Indian hedgehog':                             'Notch/Hedgehog',
    'Signaling by Steroids':                                    'Lipid/Steroid',
    'Signaling by Retinoic Acid':                               'Lipid/Steroid',
    'Signaling by Retinoid acid':                               'Lipid/Steroid',
    'Signaling by Prostaglandin':                               'Lipid/Steroid',
    'Signaling by Lysophosphatidic acid receptor':              'Lipid/Steroid',
    'Signaling by Arachidonoylglycerol':                        'Lipid/Steroid',
    'Signaling by Cholesterol/Desmosterol':                     'Lipid/Steroid',
    'Signaling by Cholesterol':                                 'Lipid/Steroid',
    'Signaling by Lipoxin/Leukotriene':                         'Lipid/Steroid',
    'Signaling by Dehydroepiandrosterone':                      'Lipid/Steroid',
    'Signaling by Estradiol':                                   'Lipid/Steroid',
    'Signaling by Progesterone':                                'Lipid/Steroid',
    'Signaling by Thromboxane':                                 'Lipid/Steroid',
    'Signaling by Choriogonadotropin':                          'Hormones',
    'Signaling by Parathyroid hormone':                         'Hormones',
    'Signaling by Growth hormone-releasing hormone':            'Hormones',
    'Signaling by Relaxin':                                     'Hormones',
    'Signaling by Angiotensinogen':                             'Hormones',
    'Signaling by Endothelin':                                  'Hormones',
    'Signaling by Angiopoietin':                                'Hormones',
    'Signaling by Kininogen':                                   'Hormones',
    'Signaling by Progonadoliberin':                            'Hormones',
    'Signaling by Insulin-like precursor':                      'Hormones',
    'Signaling by Prolactin':                                   'Hormones',
    'Signaling by Transferrin':                                 'Metabolic/Other',
    'Signaling by Apolipoprotein':                              'Metabolic/Other',
    'Signaling by Amyloid-beta precursor protein':              'Metabolic/Other',
    'Signaling by Amyloid-like protein':                        'Metabolic/Other',
    'Signaling by Plasminogen Activator':                       'Metabolic/Other',
    'Signaling by RET receptors':                               'Metabolic/Other',
    'Signaling by Ectonucleoside':                              'Metabolic/Other',
    'Signaling by Annexin':                                     'Metabolic/Other',
    'Signaling by Galectin':                                    'Metabolic/Other',
    'Signaling by Syndecan':                                    'Metabolic/Other',
    'Signaling by Midkine':                                     'Metabolic/Other',
    'Signaling by Pleiotrophin':                                'Metabolic/Other',
    'Signaling by von Willebrand factor':                       'Metabolic/Other',
    'Signaling by Prosaposin':                                  'Metabolic/Other',
    'Signaling by Selectin':                                    'Metabolic/Other',
    'Signaling by Cathepsin':                                   'Metabolic/Other',
    'Cell adhesion by Phospholipase':                           'Metabolic/Other',
    'Adhesion by Prothrombin':                                  'Metabolic/Other',
    'Adhesion by Fibribogen':                                   'Metabolic/Other',
    'Signaling by Humanin':                                     'Metabolic/Other',
    'Signaling by Growth arrest':                               'Metabolic/Other',
}

# Add _Claude suffixed versions
CATEGORY_MAP_FULL = {k: v for k, v in CATEGORY_MAP.items()}
for k, v in list(CATEGORY_MAP.items()):
    CATEGORY_MAP_FULL[k + '_Claude'] = v

liana['broad_category'] = liana['classification'].map(CATEGORY_MAP_FULL).fillna('Other')

print(f"\nBroad category distribution:")
print(liana['broad_category'].value_counts().to_string())

# ── SAVE ANNOTATION FILES ─────────────────────────────────────────────────────

liana[['ligand', 'receptor', 'lr_pair', 'classification', 'broad_category']].to_csv(
    '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv', index=False
)
print(f"\n✓ Saved: cpdb_lr_annotations.csv")

lr_func_dict = dict(zip(liana['lr_pair'], liana['broad_category']))

FUNC_COLORS = {
    'Synaptic adhesion':      '#2ecc71',
    'Teneurin/Latrophilin':   '#27ae60',
    'Glutamate signaling':    '#e74c3c',
    'GABA signaling':         '#9b59b6',
    'Glycine signaling':      '#c0392b',
    'Cholinergic signaling':  '#8e44ad',
    'Monoamine signaling':    '#e67e22',
    'Neuropeptide signaling': '#f39c12',
    'Axon guidance':          '#3498db',
    'Neurotrophic':           '#2980b9',
    'Growth factor signaling':'#1abc9c',
    'Wnt signaling':          '#16a085',
    'Notch/Hedgehog':         '#d35400',
    'ECM/Adhesion':           '#95a5a6',
    'Gap junction':           '#7f8c8d',
    'Immune/Inflammatory':    '#e74c3c',
    'Lipid/Steroid':          '#f1c40f',
    'Hormones':               '#e91e63',
    'Metabolic/Other':        '#a0522d',
    'Other':                  '#bdc3c7',
}

FUNC_ORDER = [
    'Synaptic adhesion', 'Teneurin/Latrophilin',
    'Glutamate signaling', 'GABA signaling', 'Glycine signaling',
    'Cholinergic signaling', 'Monoamine signaling', 'Neuropeptide signaling',
    'Axon guidance', 'Neurotrophic',
    'Growth factor signaling', 'Wnt signaling', 'Notch/Hedgehog',
    'ECM/Adhesion', 'Gap junction',
    'Immune/Inflammatory', 'Lipid/Steroid', 'Hormones',
    'Metabolic/Other', 'Other',
]

out = Path('/scratch/easmit31/cell_cell/lr_functional_annotations.py')
with open(out, 'w') as f:
    f.write('"""\n')
    f.write('LR functional annotations derived from CellPhoneDB classification column.\n')
    f.write('Auto-generated by build_cpdb_annotations.py\n')
    f.write('_Claude suffix = manually annotated, not directly from CellPhoneDB.\n')
    f.write('Import with: from lr_functional_annotations import LR_FUNCTIONS, FUNC_COLORS, FUNC_ORDER\n')
    f.write('"""\n\n')
    f.write('LR_FUNCTIONS = {\n')
    for lr, cat in sorted(lr_func_dict.items()):
        f.write(f'    {repr(lr)}: {repr(cat)},\n')
    f.write('}\n\n')
    f.write('FUNC_ORDER = [\n')
    for cat in FUNC_ORDER:
        f.write(f'    {repr(cat)},\n')
    f.write(']\n\n')
    f.write('FUNC_COLORS = {\n')
    for cat, color in FUNC_COLORS.items():
        f.write(f'    {repr(cat)}: {repr(color)},\n')
    f.write('}\n')

print(f"✓ Saved: {out}")
