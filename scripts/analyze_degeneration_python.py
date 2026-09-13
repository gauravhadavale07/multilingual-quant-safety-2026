import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

arms = [
    ("Qwen3-4B", "bf16", "outputs/tier1_baseline/qwen_bf16_v4/scored.jsonl"),
    ("Qwen3-4B", "nf4", "outputs/tier1_baseline/qwen_nf4_v1/scored.jsonl"),
    ("Llama-3.1-8B", "bf16", "outputs/tier1_baseline/llama_bf16_v2/scored.jsonl"),
    ("Llama-3.1-8B", "nf4", "outputs/tier1_baseline/llama_nf4_v2/scored.jsonl"),
]

def calc_degen(text):
    words = text.strip().split()
    n = len(words)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0
    d1 = len(set(words)) / n
    d2 = len(set(f"{words[i]}_{words[i+1]}" for i in range(n-1))) / max(1, n-1)
    fourgrams = [tuple(words[i:i+4]) for i in range(n-3)]
    rep4 = (len(fourgrams) - len(set(fourgrams))) / max(1, len(fourgrams))
    loop = int(rep4 > 0.35 or (n >= 500 and rep4 > 0.20))
    return n, d1, d2, rep4, loop

rows = []
for model, prec, p in arms:
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            resp = r.get('raw_response', '')
            length, d1, d2, rep4, loop = calc_degen(resp)
            
            audit_flag = int(bool(r.get('manual_review_required')))
            wg_v = r.get('wildguard_verdict')
            wg_comp = int(wg_v in ('COMPLIANCE', 'NON_REFUSAL'))
            wg_ref = int(wg_v == 'REFUSAL')
            
            rows.append({
                'model': model,
                'precision': prec,
                'tier': r.get('resource_tier'),
                'language': r.get('iso_code'),
                'length': length,
                'distinct1': d1,
                'distinct2': d2,
                'rep4': rep4,
                'loop': loop,
                'audit_flag': audit_flag,
                'wg_compliance': wg_comp,
                'wg_refusal': wg_ref
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} records.")

print("\n=== BIVARIATE CORRELATIONS WITH AUDIT FLAGGING (N=12,600) ===")
for m in ['length', 'distinct1', 'distinct2', 'rep4', 'loop']:
    r_aud, p_aud = stats.pearsonr(df[m], df['audit_flag'])
    r_comp, p_comp = stats.pearsonr(df[m], df['wg_compliance'])
    print(f"{m:<12}: corr with audit_flag = {r_aud:+.3f} (p={p_aud:.2e}) | corr with wg_compliance = {r_comp:+.3f} (p={p_comp:.2e})")

# Logistic regression using sklearn
# Predicting audit_flag from rep4, length, distinct1, is_nf4, is_lrl, is_llama
df['z_rep4'] = stats.zscore(df['rep4'])
df['z_length'] = stats.zscore(df['length'])
df['z_d1'] = stats.zscore(df['distinct1'])
df['is_nf4'] = (df['precision'] == 'nf4').astype(int)
df['is_lrl'] = (df['tier'] == 'LRL').astype(int)
df['is_llama'] = (df['model'] == 'Llama-3.1-8B').astype(int)

features = ['z_rep4', 'z_length', 'z_d1', 'is_nf4', 'is_lrl', 'is_llama']
X = df[features].values
y = df['audit_flag'].values

clf = LogisticRegression(penalty=None, max_iter=1000)
clf.fit(X, y)

# Approximate SE and p-values from Fisher Information matrix
pred_probs = clf.predict_proba(X)[:, 1]
W = pred_probs * (1 - pred_probs)
X_design = np.column_stack([np.ones(len(X)), X])
V = np.linalg.inv(X_design.T @ (X_design * W[:, None]))
se = np.sqrt(np.diag(V))
coefs = np.concatenate([clf.intercept_, clf.coef_[0]])
z_scores = coefs / se
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
odds_ratios = np.exp(coefs)

names = ['Intercept'] + features
res_df = pd.DataFrame({
    'Feature': names,
    'Coef': coefs,
    'SE': se,
    'z': z_scores,
    'p_value': p_values,
    'Odds_Ratio': odds_ratios,
    'OR_2.5%': np.exp(coefs - 1.96 * se),
    'OR_97.5%': np.exp(coefs + 1.96 * se)
})

print("\n=== MULTIVARIATE LOGISTIC REGRESSION: Predicting Audit Flagging ===")
print(res_df.to_string(index=False))

# Also predict WildGuard compliance (apparent safety collapse)
y_comp = df['wg_compliance'].values
clf_comp = LogisticRegression(penalty=None, max_iter=1000)
clf_comp.fit(X, y_comp)
pred_probs_c = clf_comp.predict_proba(X)[:, 1]
W_c = pred_probs_c * (1 - pred_probs_c)
V_c = np.linalg.inv(X_design.T @ (X_design * W_c[:, None]))
se_c = np.sqrt(np.diag(V_c))
coefs_c = np.concatenate([clf_comp.intercept_, clf_comp.coef_[0]])
z_c = coefs_c / se_c
p_c = 2 * (1 - stats.norm.cdf(np.abs(z_c)))
res_comp = pd.DataFrame({
    'Feature': names,
    'Coef': coefs_c,
    'SE': se_c,
    'z': z_c,
    'p_value': p_c,
    'Odds_Ratio': np.exp(coefs_c),
    'OR_2.5%': np.exp(coefs_c - 1.96 * se_c),
    'OR_97.5%': np.exp(coefs_c + 1.96 * se_c)
})
print("\n=== MULTIVARIATE LOGISTIC REGRESSION: Predicting WildGuard Compliance ===")
print(res_comp.to_string(index=False))
