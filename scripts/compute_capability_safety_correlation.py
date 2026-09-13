import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv('analysis/final_outputs/glmm_confirmatory_dataset.csv')

# Group by model_name, precision, task_type, language, resource_tier
means = df.groupby(['model_name', 'task_type', 'language', 'resource_tier', 'precision'])['success'].mean().unstack('precision')
means['delta_pp'] = (means['int4_nf4'] - means['bf16']) * 100

# Pivot so task_type is column
piv = means['delta_pp'].unstack('task_type').reset_index()
piv.rename(columns={'control': 'delta_capability_pp', 'safety': 'delta_safety_pp'}, inplace=True)

print("=== PER-LANGUAGE CAPABILITY VS SAFETY DELTAS ===")
print(piv.to_string(index=False))

print("\n=== CORRELATION ANALYSIS ===")
for model in piv['model_name'].unique():
    sub = piv[piv['model_name'] == model]
    
    # All languages
    r_all, p_all = stats.pearsonr(sub['delta_capability_pp'], sub['delta_safety_pp'])
    rho_all, sp_all = stats.spearmanr(sub['delta_capability_pp'], sub['delta_safety_pp'])
    print(f"\nModel: {model} (All 10 languages):")
    print(f"  Pearson r = {r_all:.3f} (p = {p_all:.3f}), Spearman rho = {rho_all:.3f} (p = {sp_all:.3f})")
    
    # Exclude LRL if Llama
    if 'Llama' in model:
        sub_nongated = sub[sub['resource_tier'] != 'LRL']
        r_ng, p_ng = stats.pearsonr(sub_nongated['delta_capability_pp'], sub_nongated['delta_safety_pp'])
        rho_ng, sp_ng = stats.spearmanr(sub_nongated['delta_capability_pp'], sub_nongated['delta_safety_pp'])
        print(f"Model: {model} (Non-gated, N=7 languages [REF, HRL, MRL]):")
        print(f"  Pearson r = {r_ng:.3f} (p = {p_ng:.3f}), Spearman rho = {rho_ng:.3f} (p = {sp_ng:.3f})")

# Also combined non-gated
ng_all = piv[(piv['model_name'] == 'Qwen3-4B') | (piv['resource_tier'] != 'LRL')]
r_comb, p_comb = stats.pearsonr(ng_all['delta_capability_pp'], ng_all['delta_safety_pp'])
print(f"\nCombined non-gated (N=17): Pearson r = {r_comb:.3f} (p = {p_comb:.3f})")
