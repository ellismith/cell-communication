#!/usr/bin/env python3
"""
Build LR functional annotations from CellPhoneDB classification column.

Three-source lookup:
  1. cpdb_uniprot  — partner_a is a uniprot ID, mapped via gene_input.csv
  2. cpdb_special  — partner_a is a special identifier (e.g. GABA_byGAD1_and_SLC32A1),
                     classification pulled directly from CellPhoneDB
  3. manual        — ligands genuinely absent from CellPhoneDB, annotated manually
                     (classification has _manual suffix)

Output columns: ligand, receptor, lr_pair, classification, broad_category, source

Usage:
    python build_cpdb_annotations.py
"""

import pandas as pd
from pathlib import Path

# ── LOAD ──────────────────────────────────────────────────────────────────────

cpdb       = pd.read_csv('/scratch/easmit31/cell_cell/CellPhoneDB-data/interaction_input_CellPhoneDB.csv')
cpdb['classification'] = cpdb['classification'].str.strip()
gene_input = pd.read_csv('/scratch/easmit31/cell_cell/CellPhoneDB-data/gene_input.csv')
liana      = pd.read_csv('/scratch/easmit31/cell_cell/CellPhoneDB-data/cellphonedb_interactions_liana_format.csv')

# ── SOURCE 1: uniprot lookup ───────────────────────────────────────────────────

prot2gene = dict(zip(gene_input['uniprot'], gene_input['gene_name']))
cpdb['gene_a'] = cpdb['partner_a'].map(prot2gene)

ligand_class_uniprot = (
    cpdb.dropna(subset=['gene_a', 'classification'])
    .groupby('gene_a')['classification']
    .agg(lambda x: x.mode()[0])
)

liana['lr_pair'] = liana['ligand'] + '|' + liana['receptor']
liana['classification'] = liana['ligand'].map(ligand_class_uniprot)
liana['source'] = liana['classification'].notna().map({True: 'cpdb_uniprot', False: None})

print(f"Total liana pairs:       {len(liana)}")
print(f"Matched via uniprot:     {liana['source'].eq('cpdb_uniprot').sum()}")

# ── SOURCE 2: special partner_a identifiers ───────────────────────────────────

special = cpdb[cpdb['gene_a'].isna() & cpdb['classification'].notna()].copy()

def parse_special_gene(partner_a):
    if '_by' not in partner_a:
        return None
    after_by = partner_a.split('_by', 1)[1]
    return after_by.split('_and_')[0]

special['gene_a_parsed'] = special['partner_a'].apply(parse_special_gene)

ligand_class_special = (
    special.dropna(subset=['gene_a_parsed', 'classification'])
    .groupby('gene_a_parsed')['classification']
    .agg(lambda x: x.mode()[0])
)

def lookup_special(ligand):
    genes = ligand.split('_')
    for g in genes:
        if g in ligand_class_special:
            return ligand_class_special[g]
    return None

unmatched_mask = liana['source'].isna()
liana.loc[unmatched_mask, 'classification'] = liana.loc[unmatched_mask, 'ligand'].apply(lookup_special)
liana.loc[unmatched_mask & liana['classification'].notna(), 'source'] = 'cpdb_special'

print(f"Matched via special id:  {liana['source'].eq('cpdb_special').sum()}")

# ── SOURCE 3: manual annotations ──────────────────────────────────────────────

MANUAL_LIGAND_CLASS = {
    # Synaptic adhesion
    'NLGN4Y':        'Signaling by Neuroligin_manual',
    'DAG1':          'Signaling by Neurexin_manual',
    'CBLN1_NRXN1':   'Signaling by Neurexin_manual',
    # Axon guidance
    'SEMA4D':        'Signaling by Semaphorin_manual',
    'UNC5A':         'Signaling by Netrin_manual',
    'CHL1':          'Adhesion by L1CAM_manual',
    'CNTN1':         'Adhesion by L1CAM_manual',
    # Notch
    'JAG1':          'Signaling by Notch_manual',
    'JAG2':          'Signaling by Notch_manual',
    # Growth Factor/Neurotrophic
    'GDNF':          'Signaling by Neurotrophin_manual',
    'NELL2':         'Signaling by Neurotrophin_manual',
    'TULP1':         'Signaling by Neurotrophin_manual',
    'NXNL1':         'Signaling by Neurotrophin_manual',
    'FGFR4':         'Signaling by Fibroblast growth factor_manual',
    'PI16':          'Signaling by Fibroblast growth factor_manual',
    'HGF':           'Signaling by Fibroblast growth factor_manual',
    'NDP':           'Signaling by Vascular endothelial growth factor_manual',
    'PTPRD':         'Signaling by RET receptors_manual',
    'PTPRS':         'Signaling by RET receptors_manual',
    # TGFβ superfamily
    'BMP15_GDF2':    'Signaling by BMP_manual',
    'BMPR1B_BMPR2':  'Signaling by BMP_manual',
    'INHBA_INHBB':   'Signaling by Inhibin/Activin_manual',
    'INHA_INHBA':    'Signaling by Inhibin/Activin_manual',
    'INHA_INHBB':    'Signaling by Inhibin/Activin_manual',
    # Neuropeptide
    'PENK':          'Signaling by Opioid_manual',
    'KISS1':         'Signaling by Neuropeptide precursor_manual',
    'MLN':           'Signaling by Neuropeptide precursor_manual',
    'NSMF':          'Signaling by Neuropeptide precursor_manual',
    'QRFP':          'Signaling by Neuropeptide precursor_manual',
    'TRH':           'Signaling by Neuropeptide precursor_manual',
    # ECM/Adhesion
    'LAMA2':         'Adhesion by Laminin_manual',
    'FBN1':          'Adhesion by Fibronectin_manual',
    'TGM2':          'Adhesion by Fibronectin_manual',
    'CSPG4':         'Adhesion by Collagen/Integrin_manual',
    'OMD':           'Adhesion by Collagen/Integrin_manual',
    'LRPAP1':        'Adhesion by Collagen/Integrin_manual',
    'MADCAM1':       'Adhesion by VCAM_manual',
    'ESAM':          'Adhesion by JAM_manual',
    'PECAM1':        'Adhesion by ICAM_manual',
    'SDC2':          'Signaling by Syndecan_manual',
    'ITGAV_ITGB1':   'Signaling by Integrin_manual',
    'ITGAV_ITGB3':   'Signaling by Integrin_manual',
    'ITGAV_ITGB5':   'Signaling by Integrin_manual',
    'FGA_FGB_FGG':   'Adhesion by Fibribogen_manual',
    # Metabolic/Other
    'MMP2':          'Signaling by Plasminogen Activator_manual',
    'PLG':           'Signaling by Plasminogen Activator_manual',
    'PROC':          'Signaling by Plasminogen Activator_manual',
    'PRNP':          'Signaling by Amyloid-beta precursor protein_manual',
    'RTN4R':         'Signaling by Amyloid-like protein_manual',
    'GP1BA':         'Signaling by von Willebrand factor_manual',
    'ALB':           'Signaling by Transferrin_manual',
    'CD320':         'Signaling by Transferrin_manual',
    'FTH1':          'Signaling by Transferrin_manual',
    'FTL':           'Signaling by Transferrin_manual',
    'HFE':           'Signaling by Transferrin_manual',
    'LCN2':          'Signaling by Transferrin_manual',
    'CLU':           'Signaling by Apolipoprotein_manual',
    'SAA1':          'Signaling by Apolipoprotein_manual',
    # Hormones/Endocrine
    'LEP':           'Signaling by Leptin_manual',
    'REN':           'Signaling by Angiotensinogen_manual',
    'FSHB':          'Signaling by Choriogonadotropin_manual',
    'LHB':           'Signaling by Choriogonadotropin_manual',
    'TSHB':          'Signaling by Choriogonadotropin_manual',
    'CGA_CGB1':      'Signaling by Choriogonadotropin_manual',
    'CGA_CGB2':      'Signaling by Choriogonadotropin_manual',
    'CGA_CGB3':      'Signaling by Choriogonadotropin_manual',
    'CGA_CGB7':      'Signaling by Choriogonadotropin_manual',
    'CGA_FSHB':      'Signaling by Choriogonadotropin_manual',
    'CGA_LHB':       'Signaling by Choriogonadotropin_manual',
    'CGA_TSHB':      'Signaling by Choriogonadotropin_manual',
    'GPHA2_GPHB5':   'Signaling by Choriogonadotropin_manual',
    # Lipid signaling
    'HEBP1':         'Signaling by Prostaglandin_manual',
    'PRXL2B':        'Signaling by Prostaglandin_manual',
    'PTGIS':         'Signaling by Prostaglandin_manual',
    # Steroid/Retinoid signaling
    'RBP4':          'Signaling by Retinoic Acid_manual',
    'SULT1A1':       'Signaling by Dehydroepiandrosterone_manual',
    'TPO':           'Signaling by Steroids_manual',
    # Immune/Inflammatory — HLA
    'HLA-E':         'Signaling by HLA_manual',
    'HLA-F':         'Signaling by HLA_manual',
    'BAG6':          'Signaling by HLA_manual',
    'CD1D':          'Signaling by HLA_manual',
    'ERVH48-1':      'Signaling by HLA_manual',
    'B2M_CD1B':      'Signaling by HLA_manual',
    # Immune/Inflammatory — Chemokines
    'ALCAM':         'Signaling by Chemokines_manual',
    'C10orf99':      'Signaling by Chemokines_manual',
    'CD200':         'Signaling by Chemokines_manual',
    'CD226':         'Signaling by Chemokines_manual',
    'CD24':          'Signaling by Chemokines_manual',
    'CD248':         'Signaling by Chemokines_manual',
    'CD34':          'Signaling by Chemokines_manual',
    'CD44':          'Signaling by Chemokines_manual',
    'CD47':          'Signaling by Chemokines_manual',
    'CD48':          'Signaling by Chemokines_manual',
    'CD52':          'Signaling by Chemokines_manual',
    'CD55':          'Signaling by Chemokines_manual',
    'CD58':          'Signaling by Chemokines_manual',
    'CD99':          'Signaling by Chemokines_manual',
    'CRTAM':         'Signaling by Chemokines_manual',
    'MPZL1':         'Signaling by Chemokines_manual',
    'PF4':           'Signaling by Chemokines_manual',
    'PPIA':          'Signaling by Chemokines_manual',
    'RNASET2':       'Signaling by Chemokines_manual',
    'SIGLEC15':      'Signaling by Chemokines_manual',
    'SPN':           'Signaling by Chemokines_manual',
    'CD177':         'Signaling by Chemokines_manual',
    # Immune/Inflammatory — TNF
    'CD274':         'Signaling by Tumor necrosis factor_manual',
    'CD276':         'Signaling by Tumor necrosis factor_manual',
    'CD70':          'Signaling by Tumor necrosis factor_manual',
    'PDCD1LG2':      'Signaling by Tumor necrosis factor_manual',
    # Immune/Inflammatory — Interleukin
    'ADIPOQ':        'Signaling by Interleukin_manual',
    'BST2':          'Signaling by Interleukin_manual',
    'CD40LG':        'Signaling by Interleukin_manual',
    'CD80':          'Signaling by Interleukin_manual',
    'CD86':          'Signaling by Interleukin_manual',
    'CLCF1':         'Signaling by Interleukin_manual',
    'CNTF':          'Signaling by Interleukin_manual',
    'CRLF2':         'Signaling by Interleukin_manual',
    'EPO':           'Signaling by Interleukin_manual',
    'FLT3LG':        'Signaling by Interleukin_manual',
    'ICOSLG':        'Signaling by Interleukin_manual',
    'KITLG':         'Signaling by Interleukin_manual',
    'LIF':           'Signaling by Interleukin_manual',
    'MST1':          'Signaling by Interleukin_manual',
    'THPO':          'Signaling by Interleukin_manual',
    'TSLP':          'Signaling by Interleukin_manual',
    'TYROBP':        'Signaling by Interleukin_manual',
    'EBI3_IL12A':    'Signaling by Interleukin_manual',
    'EBI3_IL27':     'Signaling by Interleukin_manual',
    'IL12A_IL12B':   'Signaling by Interleukin_manual',
    'IL12B_IL23A':   'Signaling by Interleukin_manual',
    'IL17A_IL17F':   'Signaling by Interleukin_manual',
    # Immune/Inflammatory — Complement
    'CD93':          'Signaling by Complement_manual',
    'SFTPD':         'Signaling by Complement_manual',
    # Immune/Inflammatory — NK/Killer cell
    'CD96':          'Signaling by Killer Cell Lectin Like Receptor K1_manual',
    'KLRB1':         'Signaling by Killer Cell Lectin Like Receptor K1_manual',
    'MICA':          'Signaling by Killer Cell Lectin Like Receptor K1_manual',
    'MICB':          'Signaling by Killer Cell Lectin Like Receptor K1_manual',
    'NCR3LG1':       'Signaling by Killer Cell Lectin Like Receptor K1_manual',
    # Immune/Inflammatory — FC receptor
    'LAIR1':         'Signaling by FC receptor_manual',
    'LCK':           'Signaling by FC receptor_manual',
    'PTPRC':         'Signaling by FC receptor_manual',
    'UBASH3B':       'Signaling by FC receptor_manual',
    # Immune/Inflammatory — other
    'PVR':           'Signaling by Poliovirus receptor_manual',
    # Teneurin
    'FLRT1_TENM2':   'Signaling by Teneurin_manual',
    'FLRT1_TENM3':   'Signaling by Teneurin_manual',
    'FLRT1_TENM4':   'Signaling by Teneurin_manual',
    'FLRT3_TENM2':   'Signaling by Teneurin_manual',
    'FLRT3_TENM3':   'Signaling by Teneurin_manual',
    'FLRT3_TENM4':   'Signaling by Teneurin_manual',
}

still_unmatched = liana['source'].isna()
liana.loc[still_unmatched, 'classification'] = liana.loc[still_unmatched, 'ligand'].map(MANUAL_LIGAND_CLASS)
liana.loc[still_unmatched & liana['classification'].notna(), 'source'] = 'manual'

# ── OVERRIDE: SLIT1/SLIT2 → Axon guidance ────────────────────────────────────
slit_mask = liana['ligand'].isin(['SLIT1', 'SLIT2'])
liana.loc[slit_mask, 'classification'] = 'Signaling by SLIT_manual'
liana.loc[slit_mask, 'source'] = 'manual'

print(f"Matched via manual dict: {liana['source'].eq('manual').sum()}")
print(f"Still unmatched:         {liana['source'].isna().sum()}")

truly_unmatched = liana[liana['source'].isna()]['ligand'].unique()
if len(truly_unmatched) > 0:
    print(f"Unmatched ligands: {truly_unmatched.tolist()}")

# ── BROAD CATEGORY MAP ────────────────────────────────────────────────────────

CATEGORY_MAP = {
    # Synaptic adhesion
    'Signaling by Neuroligin':                                  'Synaptic adhesion',
    'Signaling by Neurexin':                                    'Synaptic adhesion',
    'Signaling by Leucine-Rich Repeat Transmembrane Neuronal Protein': 'Synaptic adhesion',
    'Signaling by Calsyntenin':                                 'Synaptic adhesion',
    'Signaling by SLIT and NTRK-like protein':                  'Synaptic adhesion',
    'Signaling by Leucine Rich Repeat And Fibronectin':         'Synaptic adhesion',
    'Adhesion by CADM':                                         'Synaptic adhesion',
    # Synapse organization
    'Signaling by Agrin':                                       'Synapse organization',
    'Adhesion by Thrombospondin':                               'Synapse organization',
    'Signaling by Thrombospondin precursor':                    'Synapse organization',
    # Teneurin/Latrophilin
    'Signaling by Teneurin':                                    'Teneurin/Latrophilin',
    # Neurotransmitters
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
    'Signaling by Melatonin':                                   'Monoamine signaling',
    'Signaling by Adenosine':                                   'Purinergic signaling',
    'Signaling by Ectonucleoside':                              'Purinergic signaling',
    # Neuropeptides
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
    # Axon guidance
    'Signaling by Ephrin':                                      'Axon guidance',
    'Signaling by Semaphorin':                                  'Axon guidance',
    'Signaling by Netrin':                                      'Axon guidance',
    'Adhesion by L1CAM':                                        'Axon guidance',
    'Signaling by SLIT':                                        'Axon guidance',
    # Growth Factor/Neurotrophic
    'Signaling by Neurotrophin':                                'Growth Factor/Neurotrophic',
    'Signaling by Brain-derived neurotrophic factor':           'Growth Factor/Neurotrophic',
    'Signaling by Beta-nerve growth factor':                    'Growth Factor/Neurotrophic',
    'Signaling by Neuregulin':                                  'Growth Factor/Neurotrophic',
    'Signaling by Pro-neuregulin':                              'Growth Factor/Neurotrophic',
    'Signaling by Fibroblast growth factor':                    'Growth Factor/Neurotrophic',
    'Signaling by Epidermal growth factor':                     'Growth Factor/Neurotrophic',
    'Signaling by Insulin-like growth factor':                  'Growth Factor/Neurotrophic',
    'Signaling by Platelet-derived growth factor':              'Growth Factor/Neurotrophic',
    'Signaling by Vascular endothelial growth factor':          'Growth Factor/Neurotrophic',
    'Signaling by RET receptors':                               'Growth Factor/Neurotrophic',
    'Signaling by Midkine':                                     'Growth Factor/Neurotrophic',
    'Signaling by Pleiotrophin':                                'Growth Factor/Neurotrophic',
    'Signaling by Betacellulin':                                'Growth Factor/Neurotrophic',
    'Signaling by Placenta growth factor':                      'Growth Factor/Neurotrophic',
    'Signaling by Reelin':                                      'Growth Factor/Neurotrophic',
    # TGFβ superfamily
    'Signaling by Transforming growth factor':                  'Growth Factor/Neurotrophic',
    'Signaling by BMP':                                         'Growth Factor/Neurotrophic',
    'Signaling by Growth differentiation factor':               'Growth Factor/Neurotrophic',
    'Signaling by Inhibin/Activin':                             'Growth Factor/Neurotrophic',
    'Signaling by NODAL':                                       'Growth Factor/Neurotrophic',
    'Signaling by Muellerian-inhibiting factor':                'Growth Factor/Neurotrophic',
    'Signaling by Left-right determination factor':             'Growth Factor/Neurotrophic',
    'Signaling by Sclerostin domain-containing protein':        'Growth Factor/Neurotrophic',
    # Wnt
    'Signaling by WNT':                                         'Wnt signaling',
    'Signaling by WNT inhibition':                              'Wnt signaling',
    'Signaling by R-spondin':                                   'Wnt signaling',
    # Notch/Hedgehog
    'Signaling by Notch':                                       'Notch/Hedgehog',
    'Signaling by Sonic hedgehog':                              'Notch/Hedgehog',
    'Signaling by Desert hedgehog':                             'Notch/Hedgehog',
    'Signaling by Indian hedgehog':                             'Notch/Hedgehog',
    # Immune/Inflammatory
    'Signaling by Chemokines':                                  'Immune/Inflammatory',
    'Signaling by Interleukin':                                 'Immune/Inflammatory',
    'Signaling by Tumor necrosis factor':                       'Immune/Inflammatory',
    'Signaling by Interferon':                                  'Immune/Inflammatory',
    'Signaling by Complement':                                  'Immune/Inflammatory',
    'Signaling by HLA':                                         'Immune/Inflammatory',
    'Signaling by Colony-Stimulating factor':                   'Immune/Inflammatory',
    'Signaling by Killer Cell Lectin Like Receptor K1':         'Immune/Inflammatory',
    'Signaling by FC receptor':                                 'Immune/Inflammatory',
    'Signaling by Lymphotactin':                                'Immune/Inflammatory',
    'Signaling by Oncostatin-M precursor':                      'Immune/Inflammatory',
    'Signaling by Poliovirus receptor':                         'Immune/Inflammatory',
    'Signaling by Galectin':                                    'Immune/Inflammatory',
    'Signaling by Selectin':                                    'Immune/Inflammatory',
    'Signaling by Pro-MHC':                                     'Immune/Inflammatory',
    # ECM/Adhesion
    'Adhesion by Collagen/Integrin':                            'ECM/Adhesion',
    'Adhesion by Fibronectin':                                  'ECM/Adhesion',
    'Adhesion by Laminin':                                      'ECM/Adhesion',
    'Adhesion by ICAM':                                         'ECM/Adhesion',
    'Adhesion by Cadherin':                                     'ECM/Adhesion',
    'Adhesion by tenascin':                                     'ECM/Adhesion',
    'Adhesion by Vitronectin':                                  'ECM/Adhesion',
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
    # Gap junction
    'Adhesion by GAP junction':                                 'Gap junction',
    # Hormones/Endocrine — peptide/protein hormones
    'Signaling by Choriogonadotropin':                          'Hormones/Endocrine',
    'Signaling by Parathyroid hormone':                         'Hormones/Endocrine',
    'Signaling by Growth hormone-releasing hormone':            'Hormones/Endocrine',
    'Signaling by Relaxin':                                     'Hormones/Endocrine',
    'Signaling by Angiotensinogen':                             'Hormones/Endocrine',
    'Signaling by Endothelin':                                  'Hormones/Endocrine',
    'Signaling by Angiopoietin':                                'Hormones/Endocrine',
    'Signaling by Kininogen':                                   'Hormones/Endocrine',
    'Signaling by Progonadoliberin':                            'Hormones/Endocrine',
    'Signaling by Insulin-like precursor':                      'Hormones/Endocrine',
    'Signaling by Prolactin':                                   'Hormones/Endocrine',
    'Signaling by Leptin':                                      'Hormones/Endocrine',
    'Signaling by Triiodothyronine':                            'Hormones/Endocrine',
    # Lipid signaling
    'Signaling by Prostaglandin':                               'Lipid signaling',
    'Signaling by Lipoxin/Leukotriene':                         'Lipid signaling',
    'Signaling by Thromboxane':                                 'Lipid signaling',
    'Signaling by Arachidonoylglycerol':                        'Lipid signaling',
    'Signaling by Lysophosphatidic acid receptor':              'Lipid signaling',
    # Steroid/Retinoid signaling
    'Signaling by Steroids':                                    'Steroid/Retinoid signaling',
    'Signaling by Retinoic Acid':                               'Steroid/Retinoid signaling',
    'Signaling by Retinoid acid':                               'Steroid/Retinoid signaling',
    'Signaling by Dehydroepiandrosterone':                      'Steroid/Retinoid signaling',
    'Signaling by Progesterone':                                'Steroid/Retinoid signaling',
    'Signaling by Estradiol':                                   'Steroid/Retinoid signaling',
    'Signaling by Cholesterol/Desmosterol':                     'Steroid/Retinoid signaling',
    'Signaling by Cholesterol':                                 'Steroid/Retinoid signaling',
    # Metabolic/Other
    'Signaling by Transferrin':                                 'Metabolic/Other',
    'Signaling by Apolipoprotein':                              'Metabolic/Other',
    'Signaling by Amyloid-beta precursor protein':              'Metabolic/Other',
    'Signaling by Amyloid-like protein':                        'Metabolic/Other',
    'Signaling by Plasminogen Activator':                       'Metabolic/Other',
    'Signaling by Annexin':                                     'Metabolic/Other',
    'Signaling by Syndecan':                                    'Metabolic/Other',
    'Signaling by von Willebrand factor':                       'Metabolic/Other',
    'Signaling by Prosaposin':                                  'Metabolic/Other',
    'Signaling by Cathepsin':                                   'Metabolic/Other',
    'Cell adhesion by Phospholipase':                           'Metabolic/Other',
    'Adhesion by Prothrombin':                                  'Metabolic/Other',
    'Adhesion by Fibribogen':                                   'Metabolic/Other',
    'Signaling by Humanin':                                     'Metabolic/Other',
    'Signaling by Growth arrest':                               'Metabolic/Other',
}

# Add _manual suffixed versions
CATEGORY_MAP_FULL = dict(CATEGORY_MAP)
for k, v in list(CATEGORY_MAP.items()):
    CATEGORY_MAP_FULL[k + '_manual'] = v

# Add SLIT override directly
CATEGORY_MAP_FULL['Signaling by SLIT_manual'] = 'Axon guidance'

liana['broad_category'] = liana['classification'].map(CATEGORY_MAP_FULL).fillna('Other')

print(f"\nBroad category distribution:")
print(liana['broad_category'].value_counts().to_string())
print(f"\nSource distribution:")
print(liana['source'].value_counts().to_string())

# ── SAVE ANNOTATION CSV ───────────────────────────────────────────────────────

out_csv = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
liana[['ligand', 'receptor', 'lr_pair', 'classification', 'broad_category', 'source']].to_csv(
    out_csv, index=False
)
print(f"\n✓ Saved: {out_csv}")

# ── SAVE lr_functional_annotations.py ────────────────────────────────────────

lr_func_dict = dict(zip(liana['lr_pair'], liana['broad_category']))

FUNC_COLORS = {
    'Synaptic adhesion':          '#2ecc71',
    'Synapse organization':       '#27ae60',
    'Teneurin/Latrophilin':       '#1a7a4a',
    'Glutamate signaling':        '#e74c3c',
    'GABA signaling':             '#9b59b6',
    'Glycine signaling':          '#c0392b',
    'Cholinergic signaling':      '#8e44ad',
    'Monoamine signaling':        '#e67e22',
    'Purinergic signaling':       '#1db954',
    'Neuropeptide signaling':     '#f39c12',
    'Axon guidance':              '#3498db',
    'Growth Factor/Neurotrophic': '#1abc9c',
    'Growth Factor/Neurotrophic':           '#0e6655',
    'Wnt signaling':              '#16a085',
    'Notch/Hedgehog':             '#d35400',
    'ECM/Adhesion':               '#95a5a6',
    'Gap junction':               '#7f8c8d',
    'Immune/Inflammatory':        '#c0392b',
    'Hormones/Endocrine':         '#e91e63',
    'Lipid signaling':            '#f1948a',
    'Steroid/Retinoid signaling': '#c0392b',
    'Metabolic/Other':            '#a0522d',
    'Other':                      '#bdc3c7',
}

FUNC_ORDER = [
    'Synaptic adhesion', 'Synapse organization', 'Teneurin/Latrophilin',
    'Glutamate signaling', 'GABA signaling', 'Glycine signaling',
    'Cholinergic signaling', 'Monoamine signaling', 'Purinergic signaling',
    'Neuropeptide signaling', 'Axon guidance', 'Growth Factor/Neurotrophic',
    'Growth Factor/Neurotrophic', 'Wnt signaling', 'Notch/Hedgehog',
    'ECM/Adhesion', 'Gap junction', 'Immune/Inflammatory',
    'Hormones/Endocrine', 'Lipid signaling', 'Steroid/Retinoid signaling',
    'Metabolic/Other', 'Other',
]

out_py = Path('/scratch/easmit31/cell_cell/lr_functional_annotations.py')
with open(out_py, 'w') as f:
    f.write('"""\n')
    f.write('LR functional annotations derived from CellPhoneDB classification column.\n')
    f.write('Auto-generated by build_cpdb_annotations.py\n')
    f.write('Sources: cpdb_uniprot, cpdb_special, manual (_manual suffix).\n')
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

print(f"✓ Saved: {out_py}")
