import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

df = pd.read_csv('analysis/final_outputs/glmm_confirmatory_dataset.csv')

# Compute language-level deltas
means = df.groupby(['model_name', 'task_type', 'language', 'resource_tier', 'precision'])['success'].mean().unstack('precision')
means['delta_pp'] = (means['int4_nf4'] - means['bf16']) * 100
piv = means['delta_pp'].unstack('task_type').reset_index()
piv.rename(columns={'control': 'delta_capability', 'safety': 'delta_safety'}, inplace=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=300, sharey=True)

tier_colors = {
    'reference': '#1a6496',
    'HRL': '#2a9d8f',
    'MRL': '#e76f51',
    'LRL': '#c62828'
}

tier_markers = {
    'reference': 'o',
    'HRL': 's',
    'MRL': '^',
    'LRL': 'D'
}

# 1. Qwen3-4B
qwen = piv[piv['model_name'] == 'Qwen3-4B']
r_qwen, p_qwen = stats.pearsonr(qwen['delta_capability'], qwen['delta_safety'])

for _, row in qwen.iterrows():
    ax1.scatter(row['delta_capability'], row['delta_safety'],
                color=tier_colors[row['resource_tier']],
                marker=tier_markers[row['resource_tier']],
                s=60, edgecolors='black', linewidth=0.6, zorder=3)
    ax1.annotate(row['language'], (row['delta_capability'], row['delta_safety']),
                 xytext=(5, 3), textcoords='offset points', fontsize=8, fontweight='bold',
                 color='#222')

# Fit line for Qwen
m_q, b_q = np.polyfit(qwen['delta_capability'], qwen['delta_safety'], 1)
x_vals_q = np.linspace(qwen['delta_capability'].min() - 1, qwen['delta_capability'].max() + 1, 50)
ax1.plot(x_vals_q, m_q * x_vals_q + b_q, color='#444', linestyle='--', linewidth=1.0, alpha=0.7)

ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax1.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax1.set_title(r"$\bf{(a)\ Qwen3-4B}$ (Verified Provenance)" + f"\n$r = {r_qwen:.2f}$ ($p = {p_qwen:.2f}$, $N=10$)", fontsize=9.5)
ax1.set_xlabel(r"Capability Change $\Delta$ (Belebele, pp)", fontsize=8.5)
ax1.set_ylabel(r"Safety Refusal Change $\Delta$ (MultiJAIL, pp)", fontsize=8.5)
ax1.grid(True, linestyle='--', alpha=0.3)

# 2. Llama-3.1-8B
llama = piv[piv['model_name'] == 'Llama-3.1-8B-Instruct']
llama_ng = llama[llama['resource_tier'] != 'LRL']
r_l_ng, p_l_ng = stats.pearsonr(llama_ng['delta_capability'], llama_ng['delta_safety'])

for _, row in llama.iterrows():
    is_lrl = (row['resource_tier'] == 'LRL')
    c = tier_colors[row['resource_tier']]
    m = tier_markers[row['resource_tier']]
    if is_lrl:
        ax2.scatter(row['delta_capability'], row['delta_safety'],
                    facecolors='none', edgecolors=c, marker=m,
                    s=70, linewidth=1.5, zorder=3, linestyle='--')
        ax2.annotate(f"{row['language']} (Gated)", (row['delta_capability'], row['delta_safety']),
                     xytext=(5, -6), textcoords='offset points', fontsize=7.5, fontstyle='italic',
                     color='#888')
    else:
        ax2.scatter(row['delta_capability'], row['delta_safety'],
                    color=c, marker=m,
                    s=60, edgecolors='black', linewidth=0.6, zorder=3)
        ax2.annotate(row['language'], (row['delta_capability'], row['delta_safety']),
                     xytext=(5, 3), textcoords='offset points', fontsize=8, fontweight='bold',
                     color='#222')

# Fit line for Llama non-gated
m_l, b_l = np.polyfit(llama_ng['delta_capability'], llama_ng['delta_safety'], 1)
x_vals_l = np.linspace(llama_ng['delta_capability'].min() - 1, llama_ng['delta_capability'].max() + 1, 50)
ax2.plot(x_vals_l, m_l * x_vals_l + b_l, color='#444', linestyle='--', linewidth=1.0, alpha=0.7)

ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax2.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax2.set_title(r"$\bf{(b)\ Llama-3.1-8B}$ (Descriptive)" + f"\nNon-gated: $r = {r_l_ng:.2f}$ ($p = {p_l_ng:.2f}$, $N=7$)", fontsize=9.5)
ax2.set_xlabel(r"Capability Change $\Delta$ (Belebele, pp)", fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.3)

# Legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tier_colors['reference'], markeredgecolor='black', markersize=7, label='Reference (en)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tier_colors['HRL'], markeredgecolor='black', markersize=7, label='HRL (zh, it, vi)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=tier_colors['MRL'], markeredgecolor='black', markersize=7, label='MRL (ar, ko, th)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=tier_colors['LRL'], markeredgecolor='black', markersize=7, label='LRL (bn, sw, jv)'),
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('fig2_capability_vs_safety_scatter.pdf', bbox_inches='tight')
plt.savefig('fig2_capability_vs_safety_scatter.png', bbox_inches='tight', dpi=300)
print("Saved fig2_capability_vs_safety_scatter.pdf and .png")
