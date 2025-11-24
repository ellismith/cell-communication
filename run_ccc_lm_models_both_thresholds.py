import pandas as pd
import numpy as np
from pathlib import Path
import glob
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

# Set paths
results_dir = Path("/scratch/easmit31/cell_cell/results/u01_per_individual")
output_dir = Path("/scratch/easmit31/cell_cell/lm_results")
output_dir.mkdir(exist_ok=True)

# Load all individual results files
all_files = glob.glob(str(results_dir / "*_results.csv"))
df_list = []

print(f"Loading {len(all_files)} result files...")
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Combine all results
df_all = pd.concat(df_list, ignore_index=True)

print(f"Total rows: {len(df_all)}")
print(f"Unique animals: {df_all['animal_id'].nunique()}")

# Filter for specific sender-receiver pair
sender = "Glutamatergic_EC"
receiver = "GABA_HIP"

df_pair = df_all[(df_all['source'] == sender) & (df_all['target'] == receiver)].copy()

print(f"\nFiltered to {sender} → {receiver}:")
print(f"Unique L-R pairs: {df_pair.groupby(['ligand_complex', 'receptor_complex']).ngroups}")
print(f"Animals: {df_pair['animal_id'].nunique()}")

# Create L-R pair identifier
df_pair['LR_pair'] = df_pair['ligand_complex'] + '_' + df_pair['receptor_complex']

# Pivot to wide format
df_wide = df_pair.pivot_table(
    index='animal_id',
    columns='LR_pair',
    values='magnitude_rank',
    aggfunc='first'
)

print(f"\nWide format shape: {df_wide.shape}")
print(f"Animals (rows): {df_wide.shape[0]}")
print(f"L-R pairs (columns): {df_wide.shape[1]}")

# Get metadata
metadata = df_pair[['animal_id', 'age', 'sex']].drop_duplicates().set_index('animal_id')
df_model = df_wide.merge(metadata, left_index=True, right_index=True, how='left')

print(f"\nFinal modeling dataframe shape: {df_model.shape}")
print(f"Columns: age, sex, + {df_model.shape[1] - 2} L-R pairs")
print(f"\nAge range: {df_model['age'].min()} - {df_model['age'].max()}")
print(f"Sex distribution:\n{df_model['sex'].value_counts()}")

# Encode sex as binary for all data upfront (0=F, 1=M)
df_model['sex_binary'] = (df_model['sex'] == 'M').astype(int)

# Get list of L-R pair columns (exclude age, sex, sex_binary)
lr_columns = [col for col in df_model.columns if col not in ['age', 'sex', 'sex_binary']]

print(f"\n{'='*60}")
print("Running ALL 621 models (no filtering)")
print(f"{'='*60}")

results_list = []
skipped_errors = 0

for i, lr_pair in enumerate(lr_columns):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(lr_columns)} models...")
    
    # Get non-null values
    data = df_model[['age', 'sex_binary', lr_pair]].dropna()
    
    if len(data) < 3:  # Need at least 3 samples for regression
        print(f"  Warning: Only {len(data)} samples for {lr_pair}, skipping")
        skipped_errors += 1
        continue
    
    try:
        # Check if there's sex variance
        has_sex_variance = data['sex_binary'].nunique() == 2
        
        if has_sex_variance:
            # Normal model with age + sex
            X = data[['age', 'sex_binary']].values
            y = data[lr_pair].values
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            results_list.append({
                'LR_pair': lr_pair,
                'n_samples': len(data),
                'has_sex_variance': True,
                'intercept': model.params[0],
                'age_coef': model.params[1],
                'age_se': model.bse[1],
                'age_tstat': model.tvalues[1],
                'age_pval': model.pvalues[1],
                'sex_coef': model.params[2],
                'sex_se': model.bse[2],
                'sex_tstat': model.tvalues[2],
                'sex_pval': model.pvalues[2],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_pval': model.f_pvalue
            })
        else:
            # Model with age only (no sex variance)
            X = data[['age']].values
            y = data[lr_pair].values
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            results_list.append({
                'LR_pair': lr_pair,
                'n_samples': len(data),
                'has_sex_variance': False,
                'intercept': model.params[0],
                'age_coef': model.params[1],
                'age_se': model.bse[1],
                'age_tstat': model.tvalues[1],
                'age_pval': model.pvalues[1],
                'sex_coef': np.nan,
                'sex_se': np.nan,
                'sex_tstat': np.nan,
                'sex_pval': np.nan,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_pval': model.f_pvalue
            })
    except Exception as e:
        print(f"  Error with {lr_pair}: {e}")
        skipped_errors += 1
        continue

# Convert to dataframe
results_df = pd.DataFrame(results_list)

print(f"\nCompleted {len(results_df)} models")
print(f"Models with sex variance: {results_df['has_sex_variance'].sum()}")
print(f"Models without sex variance (age only): {(~results_df['has_sex_variance']).sum()}")
print(f"Errors/skipped: {skipped_errors}")
print(f"Mean samples per model: {results_df['n_samples'].mean():.1f}")

# Apply FDR correction
results_df['age_qval'] = multipletests(results_df['age_pval'], method='fdr_bh')[1]

# Only correct sex p-values for models that have sex variance
sex_pvals = results_df['sex_pval'].dropna()
if len(sex_pvals) > 0:
    sex_qvals = multipletests(sex_pvals, method='fdr_bh')[1]
    results_df.loc[sex_pvals.index, 'sex_qval'] = sex_qvals
else:
    results_df['sex_qval'] = np.nan

# Save results
output_file = output_dir / f"{sender}_to_{receiver}_lm_results_all_621.csv"
results_df.to_csv(output_file, index=False)

print(f"\nSignificant associations (p < 0.05):")
print(f"Age: {(results_df['age_pval'] < 0.05).sum()}")
print(f"Sex: {(results_df['sex_pval'] < 0.05).sum()}")

print(f"\nSignificant associations (FDR < 0.05):")
print(f"Age: {(results_df['age_qval'] < 0.05).sum()}")
print(f"Sex: {(results_df['sex_qval'] < 0.05).sum()}")

print(f"\nTop 10 age-associated L-R pairs:")
print(results_df.nsmallest(10, 'age_pval')[['LR_pair', 'age_coef', 'age_pval', 'age_qval', 'has_sex_variance']])

print(f"\nTop 10 sex-associated L-R pairs (where sex tested):")
sex_tested = results_df[results_df['has_sex_variance']].copy()
if len(sex_tested) > 0:
    print(sex_tested.nsmallest(10, 'sex_pval')[['LR_pair', 'sex_coef', 'sex_pval', 'sex_qval']])

print(f"\nResults saved to: {output_file}")
