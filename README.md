# Quantization and Multilingual Safety in Low-Resource Languages
## Capability Degradation, Safety Behavior, and Evaluation Artifacts
*NeurIPS 2026 Short Paper Track (Anonymous Submission)*

This repository contains the full reproduction package, raw evaluation outputs, confirmatory regression datasets, and blinded human adjudication artifacts for our pre-registered audit.

---

## Quick Start (One-Click Reproduction)

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the master reproduction script**:
```bash
python reproduce_all.py
```
This script executes the entire evaluation pipeline and verifies every statistical claim, table, and figure in the paper.

---

## Individual Reproduction Steps

### 1. Table 1: Inter-Rater Reliability & Adjudication Validation
Computes Cohen's kappa ($\kappa$) and percent agreement across AI evaluators and the blinded frozen human review dataset ($N=200$):
```bash
python scripts/calculate_final_irr.py
```
- **Panel A (3-Model AI Panel, $N=85$)**: Majority AI Consensus achieves $\kappa = 0.74$ (84.7% agreement).
- **Panel B (Full Frozen Human Adjudication, $N=200$)**: Qwen-14B achieves $\kappa = 0.45$ (69.5% agreement).

### 2. Primary Findings: Capability Loss vs. Refusal Correlations
Computes the language-level correlation between capability changes (Belebele reading comprehension) and safety refusal changes (MultiJAIL):
```bash
python scripts/compute_capability_safety_correlation.py
```
- Combined non-gated ($N=17$): Pearson $r = 0.010$ ($p = 0.97$).
- Llama-3.1-8B non-gated ($N=7$): Pearson $r = -0.221$ ($p = 0.63$).
- Qwen3-4B ($N=10$): Pearson $r = 0.151$ ($p = 0.68$).

### 3. Tables 7 & 8: Quantitative Degeneration Diagnostics
Computes token-level repetition metrics (Distinct-1, Rep-4, length) across 12,600 generations and runs multivariate logistic regressions predicting automated audit flags:
```bash
python scripts/analyze_degeneration_python.py
```

### 4. Publication Figures
Generates publication-quality vector PDFs and PNGs:
- **Figure 1** (Double Dissociation Bar Plots):
  ```bash
  python scripts/create_publication_figures.py
  ```
- **Figure 2** (Capability vs. Safety Scatter Plot):
  ```bash
  python scripts/generate_scatter_figure.py
  ```

### 5. Manuscript Numerical Consistency Audit
Audits all reported statistics against the compiled manuscript:
```bash
python scripts/verify_all_numbers.py
```

---

## Repository Structure

```text
├── README.md                               <-- This documentation
├── requirements.txt                        <-- Python dependencies
├── reproduce_all.py                        <-- One-click master replication script
├── neurips2026_short_paper.pdf             <-- Compiled 8-page manuscript
├── analysis/
│   ├── final_outputs/
│   │   ├── glmm_results.json               <-- Official GLMM coefficients & variances
│   │   └── glmm_confirmatory_dataset.csv   <-- Full 46,487-row confirmatory dataset
│   └── adjudication_ai/
│       ├── human_review_packet_blinded_FINAL.csv <-- Frozen N=200 human adjudication
│       ├── human_validation_final_200.csv        <-- Detailed Stage 2 review records
│       └── human_validation_original.csv         <-- Original Stage 1 review records
├── scripts/
│   ├── calculate_final_irr.py              <-- Table 1 IRR computation
│   ├── compute_capability_safety_correlation.py <-- Correlation analysis
│   ├── analyze_degeneration_python.py      <-- Degeneration & regression diagnostics
│   ├── create_publication_figures.py       <-- Figure 1 generator
│   ├── generate_scatter_figure.py          <-- Figure 2 generator
│   └── verify_all_numbers.py               <-- Manuscript number verification
└── outputs/
    ├── tier1_baseline/                     <-- Model generations and scored outputs
    │   ├── llama_nf4_v2/scored.jsonl       <-- Provenance-verified Llama NF4 run
    │   ├── llama_bf16_v2/scored.jsonl      <-- Baseline Llama BF16 run
    │   ├── llama_nf4_v1/scored.jsonl       <-- Historical Llama NF4 run
    │   ├── qwen_bf16_v4/scored.jsonl       <-- Baseline Qwen BF16 run
    │   └── qwen_nf4_v1/scored.jsonl        <-- Qwen NF4 run
    └── audits/                             <-- Degeneration and scoring audit logs
```

---

## Provenance and Governance Note
As described in Section 3 of the manuscript:
- The apparent $-58.9$ pp refusal collapse in Llama-3.1-8B low-resource languages is quarantined behind a **Governance Gate** due to severe repetition loops inflating automated non-refusal flags (with 0.0% actionable harm).
- Confirmatory GLMM inference is conducted on the $N = 46,487$ non-gated observations.
- All evaluation records and frozen human validation adjudications ($N=200$) are permanently preserved with cryptographic hashes.
