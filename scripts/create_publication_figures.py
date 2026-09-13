"""Generate publication-ready NeurIPS 2026 figures for the paper.
Produces vector PDF and high-DPI PNG formats with clean typography and palettes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

TIER_LABELS = ["Reference\n(en)", "High-Resource\n(zh, it, vi)", "Medium-Resource\n(ar, ko, th)", "Low-Resource\n(bn, sw, jv)"]
TIERS = ["REF", "HRL", "MRL", "LRL"]

# Data from Table 2 (Belebele Capability)
qwen_belebele_bf16 = [84.3, 81.4, 78.8, 58.8]
qwen_belebele_nf4  = [82.5, 72.6, 71.9, 52.0]  # Deltas: -1.8, -8.8, -6.9, -6.8

llama_belebele_bf16 = [88.9, 84.1, 76.3, 62.3]
llama_belebele_nf4  = [79.5, 74.5, 64.7, 48.6]  # Deltas: -9.4, -9.6, -11.6, -13.7

# Data from Table 3 (MultiJAIL Safety Refusal)
qwen_multijail_bf16 = [98.5, 88.2, 83.5, 25.0]
qwen_multijail_nf4  = [96.6, 84.4, 85.7, 33.3]  # Deltas: -1.9, -3.8, +2.3, +8.3

llama_multijail_bf16 = [98.6, 90.5, 89.2, 80.0]
llama_multijail_nf4_clean = [99.5, 89.2, 87.8, np.nan] # LRL is gated!
llama_multijail_nf4_artifact = [np.nan, np.nan, np.nan, 13.0] # Replicated LRL collapse (v2)

def make_figure_1_double_dissociation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 1.85), dpi=300)
    fig.subplots_adjust(wspace=0.28, bottom=0.25, top=0.88, left=0.08, right=0.98)

    x = np.arange(len(TIERS))
    width = 0.18

    # ── Panel A: Capability (Belebele) ──
    # Qwen BF16 / NF4
    ax1.bar(x - 1.5*width, qwen_belebele_bf16, width, label="Qwen3-4B (BF16)", color="#4575b4", alpha=0.9)
    ax1.bar(x - 0.5*width, qwen_belebele_nf4,  width, label="Qwen3-4B (NF4)",  color="#91bfdb", alpha=0.9, hatch="//")
    # Llama BF16 / NF4
    ax1.bar(x + 0.5*width, llama_belebele_bf16, width, label="Llama-3.1-8B (BF16)", color="#d73027", alpha=0.9)
    ax1.bar(x + 1.5*width, llama_belebele_nf4,  width, label="Llama-3.1-8B (NF4)",  color="#fc8d59", alpha=0.9, hatch="\\\\")

    ax1.set_ylabel("Belebele Accuracy (%)")
    ax1.set_title("(a) Capability Degradation (H2 Supported)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(TIER_LABELS, fontsize=7.5)
    ax1.set_ylim(30, 105)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Monotonic degradation annotation for Llama
    ax1.annotate("Monotonic gap\nwidening: -13.7 pp", xy=(3 + 1.5*width, 49), xytext=(2.1, 34),
                 arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
                 fontsize=7, color="#c62828", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffebee", ec="#c62828", lw=0.8))

    # ── Panel B: Safety Refusal (MultiJAIL) ──
    ax2.bar(x - 1.5*width, qwen_multijail_bf16, width, label="Qwen3-4B (BF16)", color="#4575b4", alpha=0.9)
    ax2.bar(x - 0.5*width, qwen_multijail_nf4,  width, label="Qwen3-4B (NF4)",  color="#91bfdb", alpha=0.9, hatch="//")
    ax2.bar(x + 0.5*width, llama_multijail_bf16, width, label="Llama-3.1-8B (BF16)", color="#d73027", alpha=0.9)
    ax2.bar(x + 1.5*width, llama_multijail_nf4_clean, width, label="Llama-3.1-8B (NF4)", color="#fc8d59", alpha=0.9, hatch="\\\\")

    # Artifact bar with dashed red border
    ax2.bar(3 + 1.5*width, llama_multijail_nf4_artifact[3], width, color="#ffebee", edgecolor="#d73027", linestyle="--", lw=1.5, hatch="xx")

    ax2.set_ylabel("Safety Refusal Rate (%)")
    ax2.set_title("(b) Safety Refusal (H1 Unsupported)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(TIER_LABELS, fontsize=7.5)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Governance gate annotation
    ax2.annotate("Governance Gate:\nReproducible degeneration /\nevaluator confound", xy=(3 + 1.5*width, 14), xytext=(1.5, 46),
                 arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2),
                 fontsize=6.5, color="#b71c1c", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffebee", ec="#b71c1c", lw=0.8))

    # Legend at bottom center across both
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=False, fontsize=7.8)

    out_pdf = FIG_DIR / "fig1_double_dissociation.pdf"
    out_png = FIG_DIR / "fig1_double_dissociation.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    
    # Also save to root directory for tectonic
    import shutil
    shutil.copy2(out_pdf, ROOT / "fig1_double_dissociation.pdf")
    shutil.copy2(out_png, ROOT / "fig1_double_dissociation.png")
    plt.close()
    print(f"Generated Figure 1 at: {out_pdf} and root")

if __name__ == "__main__":
    make_figure_1_double_dissociation()
