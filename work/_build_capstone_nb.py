import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Capstone — Refresh / Content Opportunity Scoring

**Lane:** Refresh / Content Opportunity Scoring
**Question:** Which content items should a limited review team open first to catch pages at risk of meaningful click decline?

This notebook is the reproducible core of the deployed paper (`docs/index.html`). It runs two analyses:

1. **Starter slice** (30,000 rows, ships in this repo) — runs locally, no credentials.
2. **Warehouse future-window** (Feb 2026 features → March 2026 outcome) — needs a Hugging Face read token.

Both use a **client-holdout split** and compare a random forest against a transparent baseline. Run top to bottom. Start with the setup cell.""")

md("## 0. Setup")
code("""# One-time: install deps + authenticate to the gated warehouse
# %pip -q install duckdb huggingface_hub pandas scikit-learn matplotlib
# Then in a terminal:  hf auth login   (paste your READ token)
import os
print("cwd:", os.getcwd())""")

md("""## 1. The decision

The output is a **ranked review queue**: a score per content item, ordered so a human reviewer spends limited capacity on the pages most likely to have meaningfully lost clicks. It is decision support — it does not claim a refresh *causes* recovery.""")

md("""## 2. Starter-slice analysis (runs locally, no credentials)

Client-holdout split, magnitude-aware label from the raw 30-day windows, random forest vs baseline.""")

code("""# Run the starter analysis (also writes work/outputs/starter_results.csv)
%run work/capstone_analysis_starter.py""")

md("""**Reading the numbers:** the baseline (a hand rule) reaches precision@50 ≈ 0.30 on held-out clients; the logistic and random-forest models reach ≈ 0.80 / 0.64 respectively, for a ~2× lift. AUC and average precision move the same direction. The `work/outputs/starter_refresh_queue.csv` file is the ranked queue.""")

md("""## 3. Warehouse future-window analysis (needs HF token)

This is the primary result. Features come from **February 2026**; the label is **March 2026** — a true future window, so nothing from March can leak into a feature.""")

code("""# Connect + run the warehouse analysis (writes work/outputs/warehouse_results.json)
%run work/explore_warehouse.py    # first: confirm schemas
%run work/capstone_analysis_warehouse.py   # then: full model + queue""")

md("""## 4. What the model finds

The strongest features are volume, position, and content age — a "visible, aging, slipping" profile — not a single magic column. The precision@K curve shows the model beats the baseline across review capacities, not just at one K.""")

md("""## 5. Validation design (why this is honest)

- **Client-holdout split** (`GroupShuffleSplit` on `client_hash_id`): pages from the same client share structure; a random row split would let the model peek at a client it already saw.
- **Leakage audit:** no March column in any feature, no `trend_direction`/`trend_pct`, IDs used for grouping only, the 90-day query table excluded (its window overlaps the label month).
- **Metrics vs base rate:** precision@K and AUC are reported next to the base rate, because a high precision can just be a high base rate.""")

md("""## 6. Ranked recommendations

The queue maps each page to a reason code and a suggested action: `refresh`, `refresh_review`, `expand`, `review_ctr`, `monitor`. See `work/outputs/warehouse_refresh_queue.csv`.""")

md("""## 7. Regenerate the paper

```bash
python work/build_paper.py
```
Reads the artifacts in `work/outputs/` and rewrites `docs/index.html` (the deployed page).""")

md("""## 8. Self-check

- [x] Question, data, baseline, model, validation, results, limitations, recommendations all present
- [x] Client-holdout split (not a random row split)
- [x] Base rate reported next to every precision@K
- [x] No client names / domains / URLs / queries anywhere
- [x] observed / directional / decision-support language
- [x] Deterministic (seed 42, fixed build + windows)""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "work/notebooks/capstone.ipynb")
print("wrote work/notebooks/capstone.ipynb with", len(cells), "cells")
