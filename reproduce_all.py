"""
Master Reproduction Script for:
Quantization and Multilingual Safety in Low-Resource Languages (NeurIPS 2026 Short Paper)

Runs all core analysis scripts and verifies exact numerical matches with the published manuscript.
"""

import subprocess
import sys
import os

SCRIPTS = [
    ("Table 1 (Inter-Rater Reliability / Kappa)", [sys.executable, "scripts/calculate_final_irr.py"]),
    ("Capability vs. Safety Correlations (r and p values)", [sys.executable, "scripts/compute_capability_safety_correlation.py"]),
    ("Tables 7 & 8 (Degeneration Diagnostics & Logit Regressions)", [sys.executable, "scripts/analyze_degeneration_python.py"]),
    ("Figure 1 (Capability & Safety Bar Plots)", [sys.executable, "scripts/create_publication_figures.py"]),
    ("Figure 2 (Capability vs. Safety Scatter Plot)", [sys.executable, "scripts/generate_scatter_figure.py"]),
    ("Paper Numerical Consistency Audit", [sys.executable, "scripts/verify_all_numbers.py"]),
]

def main():
    print("=" * 70)
    print("REPRODUCING ALL RESULTS: NeurIPS 2026 Short Paper")
    print("=" * 70)
    all_passed = True

    for name, cmd in SCRIPTS:
        print(f"\n>>> Running: {name}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[PASS] {name}")
            lines = res.stdout.strip().splitlines()
            for l in lines[-6:]:
                print(f"    {l}")
        else:
            print(f"[FAIL] {name}")
            print(res.stderr)
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL REPRODUCTION STEPS COMPLETED AND VERIFIED SUCCESSFULLY!")
    else:
        print("SOME STEPS ENCOUNTERED ISSUES. Please see output above.")
    print("=" * 70)

if __name__ == "__main__":
    main()
