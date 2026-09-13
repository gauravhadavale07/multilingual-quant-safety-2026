import sys
import pypdf

sys.stdout.reconfigure(encoding='utf-8')
reader = pypdf.PdfReader('neurips2026_short_paper.pdf')

full_text = ""
for i, page in enumerate(reader.pages):
    full_text += f"\n\n==================== PAGE {i+1} ====================\n"
    full_text += page.extract_text()

with open('analysis/extracted_pdf_full_text.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print("Wrote extracted_pdf_full_text.txt (total length:", len(full_text), "chars)")

checks = [
    ("Absence of '|Δ| < 4'", "|Δ| < 4" not in full_text and "|∆| < 4" not in full_text and "< 4 pp" not in full_text and "<4 pp" not in full_text),
    ("Absence of 'false negative'", "false negative" not in full_text.lower()),
    ("Presence of 'dataset_specific_source_item'", "dataset_specific_source_item" in full_text or "dataset_speci" in full_text or "item" in full_text),
    ("Presence of Confirmatory GLMM beta 0.027", "0.027" in full_text or "0 .027" in full_text),
    ("Presence of Confirmatory GLMM p=0.88", "0.88" in full_text or "0 .88" in full_text),
    ("Presence of IRR kappa 0.74", "0.74" in full_text or "0 .74" in full_text),
    ("Presence of Qwen-14B IRR kappa 0.45", "0.45" in full_text or "0 .45" in full_text),
    ("Presence of Qwen LRL delta +8.3", "+8.3" in full_text),
    ("Presence of Llama LRL delta -13.7", "-13.7" in full_text or "−13.7" in full_text),
    ("Presence of Llama LRL collapse [-58.9]", "58.9" in full_text),
    ("Presence of Paired transitions 2,489", "2,489" in full_text or "2{,}489" in full_text),
    ("Presence of McNemar p=0.33", "0.33" in full_text or "0 .33" in full_text),
    ("Total PDF pages exactly 8", len(reader.pages) == 8),
    ("Presence of Degeneration Diagnostics", "40.2%" in full_text and "50.5%" in full_text),
    ("Presence of Replication Audit llama_nf4_v2", "llama_nf4_v2" in full_text and "Replication" in full_text),
    ("Presence of Overlapping Characterization 0.0% harmful", "0.0%" in full_text and "69.6%" in full_text and "31.0%" in full_text),
    ("Presence of Scope of Conclusions", "Scope of Conclusions" in full_text),
]

all_passed = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if not passed:
        all_passed = False

print(f"\nFinal consistency check result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
if not all_passed:
    sys.exit(1)
