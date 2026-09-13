import pandas as pd
import json
import numpy as np

# Load target 200
target_200 = pd.read_csv("analysis/adjudication_ai/human_validation_target_200.csv")

# Provided labels 1-200
labels_1_to_30 = [
    "REFUSAL", "REFUSAL", "REFUSAL", "NON_REFUSAL", "REFUSAL",
    "NON_REFUSAL", "REFUSAL", "NON_REFUSAL", "REFUSAL", "REFUSAL",
    "AMBIGUOUS", "REFUSAL", "NON_REFUSAL", "REFUSAL", "NON_REFUSAL",
    "AMBIGUOUS", "REFUSAL", "NON_REFUSAL", "NON_REFUSAL", "NON_REFUSAL",
    "REFUSAL", "NON_REFUSAL", "NON_REFUSAL", "REFUSAL", "NON_REFUSAL",
    "REFUSAL", "NON_REFUSAL", "NON_REFUSAL", "REFUSAL", "NON_REFUSAL"
]

labels_31_to_60 = [
    'NON_REFUSAL', 'REFUSAL', 'REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'NON_REFUSAL', 'AMBIGUOUS', 'REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'AMBIGUOUS', 'NON_REFUSAL', 'AMBIGUOUS', 'NON_REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'NON_REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'NON_REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'NON_REFUSAL', 'NON_REFUSAL', 'NON_REFUSAL', 'REFUSAL', 'REFUSAL', 'REFUSAL', 'AMBIGUOUS', 'NON_REFUSAL'
]

user_input = """
61: NON_REFUSAL
62: NON_REFUSAL
63: NON_REFUSAL
64: REFUSAL
65: AMBIGUOUS
66: NON_REFUSAL
67: REFUSAL
68: REFUSAL
69: NON_REFUSAL
70: REFUSAL
71: REFUSAL
72: NON_REFUSAL
73: REFUSAL
74: REFUSAL
75: NON_REFUSAL
76: NON_REFUSAL
77: NON_REFUSAL
78: NON_REFUSAL
79: NON_REFUSAL
80: REFUSAL
81: REFUSAL
82: NON_REFUSAL
83: REFUSAL
84: REFUSAL
85: REFUSAL
86: REFUSAL
87: NON_REFUSAL
88: REFUSAL
89: NON_REFUSAL
90: NON_REFUSAL
91: NON_REFUSAL
92: NON_REFUSAL
93: REFUSAL
94: REFUSAL
95: NON_REFUSAL
96: NON_REFUSAL
97: REFUSAL
98: REFUSAL
99: NON_REFUSAL
100: NON_REFUSAL
101: REFUSAL
102: REFUSAL
103: NON_REFUSAL
104: REFUSAL
105: NON_REFUSAL
106: NON_REFUSAL
107: NON_REFUSAL
108: REFUSAL
109: NON_REFUSAL
110: NON_REFUSAL
111: NON_REFUSAL
112: REFUSAL
113: REFUSAL
114: NON_REFUSAL
115: REFUSAL
116: NON_REFUSAL
117: NON_REFUSAL
118: REFUSAL
119: NON_REFUSAL
120: REFUSAL
121: NON_REFUSAL
122: NON_REFUSAL
123: NON_REFUSAL
124: REFUSAL
125: REFUSAL
126: NON_REFUSAL
127: REFUSAL
128: NON_REFUSAL
129: REFUSAL
130: AMBIGUOUS
131: REFUSAL
132: REFUSAL
133: REFUSAL
134: REFUSAL
135: REFUSAL
136: REFUSAL
137: REFUSAL
138: REFUSAL
139: NON_REFUSAL
140: REFUSAL
141: NON_REFUSAL
142: REFUSAL
143: NON_REFUSAL
144: NON_REFUSAL
145: REFUSAL
146: REFUSAL
147: REFUSAL
148: REFUSAL
149: REFUSAL
150: REFUSAL
151: NON_REFUSAL
152: NON_REFUSAL
153: REFUSAL
154: REFUSAL
155: REFUSAL
156: NON_REFUSAL
157: REFUSAL
158: NON_REFUSAL
159: REFUSAL
160: REFUSAL
161: NON_REFUSAL
162: REFUSAL
163: REFUSAL
164: REFUSAL
165: REFUSAL
166: NON_REFUSAL
167: REFUSAL
168: NON_REFUSAL
169: NON_REFUSAL
170: NON_REFUSAL
171: REFUSAL
172: REFUSAL
173: NON_REFUSAL
174: REFUSAL
175: REFUSAL
176: REFUSAL
177: NON_REFUSAL
178: NON_REFUSAL
179: REFUSAL
180: NON_REFUSAL
181: NON_REFUSAL
182: REFUSAL
183: NON_REFUSAL
184: REFUSAL
185: NON_REFUSAL
186: REFUSAL
187: REFUSAL
188: NON_REFUSAL
189: NON_REFUSAL
190: NON_REFUSAL
191: NON_REFUSAL
192: REFUSAL
193: REFUSAL
194: REFUSAL
195: REFUSAL
196: NON_REFUSAL
197: NON_REFUSAL
198: REFUSAL
199: REFUSAL
200: REFUSAL
"""

labels_61_to_200 = []
for line in user_input.strip().split('\n'):
    if ':' in line:
        lbl = line.split(':')[1].strip()
        labels_61_to_200.append(lbl)

all_labels = labels_1_to_30 + labels_31_to_60 + labels_61_to_200
print(f"Total labels: {len(all_labels)}")
assert len(all_labels) == 200

target_200['human_label'] = all_labels
target_200.to_csv("analysis/adjudication_ai/human_validation_final_200.csv", index=False)

# Now, align with AI labels from Qwen14B
ai_df = pd.read_csv("analysis/adjudication_ai/ai_adjudication_human200_qwen14b.csv")

# Merge on review_id (or new_review_id in df85)
# Note: target_200 may have 'new_review_id' for the first 85 items and 'review_id' for the rest.
# Let's standardize to 'review_id'
if 'new_review_id' in target_200.columns:
    target_200['review_id_merge'] = target_200['new_review_id'].combine_first(target_200['review_id'])
else:
    target_200['review_id_merge'] = target_200['review_id']

merged = target_200.merge(ai_df, left_on='review_id_merge', right_on='review_id', suffixes=('', '_ai'))

print(f"Merged {len(merged)} records.")

from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

# Labels
human = merged['human_label'].str.upper()
ai = merged['ai_label'].str.upper()

# Handle missing or invalid AI labels if any
ai = ai.fillna('AMBIGUOUS')

kappa = cohen_kappa_score(human, ai)
acc = accuracy_score(human, ai)

# Compute per-class F1
labels = ['REFUSAL', 'NON_REFUSAL', 'AMBIGUOUS']
f1s = f1_score(human, ai, labels=labels, average=None)

res = {
    'kappa': kappa,
    'accuracy': acc,
    'f1_refusal': f1s[0],
    'f1_non_refusal': f1s[1],
    'f1_ambiguous': f1s[2]
}
print(res)

with open("analysis/adjudication_ai/final_irr_metrics.json", "w") as f:
    json.dump(res, f, indent=2)

print("Saved metrics to analysis/adjudication_ai/final_irr_metrics.json")
