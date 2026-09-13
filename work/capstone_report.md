# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** (your name)
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/PREK-X/flyrank-internship
- **Date:** September 2026

## 1. Problem framing

Which content items should a limited review team open first, to catch pages at risk of a meaningful click decline? Unit of analysis = one content item (page) per pseudonymized client. Output = a ranked queue with a score, reason codes, and a suggested action. Action = a human reviewer refreshes / expands / protects / monitors / prunes. Cost of a wrong call = wasted review time (false positive) or a compounding visibility loss (false negative). ML helps because the decision depends on non-linear interactions (volume × position × CTR × freshness × depth) that a fixed rule can't weigh.

## 2. Data safety

Used: `fact_content_daily_performance` (Feb + Mar 2026 partitions) and `dim_content`, from the gated `FlyRank/internship-warehouse` release (build `v20260703`). Excluded: client names/domains/URLs/queries/titles (not in the release), `trend_direction`/`trend_pct` (label-derived → leakage), `gsc_data_available=FALSE` rows, pages with < 5 Feb clicks (noise floor), and the 90-day query table (window overlaps the label month). No client-identifying detail appears anywhere in `work/`. IDs are pseudonyms used for grouping/splitting only.

## 3. Baseline

A transparent weighted rule on February signals: `0.35·visibility + 0.30·staleness + 0.20·weak-position + 0.15·low-CTR`, each a percentile rank. It encodes the same "stale-but-visible, poorly-positioned, under-clicking" intuition the model is asked to beat, and uses only the feature window — a fair comparison.

## 4. Model / analysis

Random forest classifier (also logistic regression for readability). Label = `clicks_feb ≥ 5 AND clicks_mar ≤ 0.7·clicks_feb` (meaningful decline). Features: Feb impressions/clicks/CTR/position/sessions/engagement/active-days (log-transformed heavy tails) + dim_content word/char count, content age, days-since-update, content type, intent. Left out on purpose: everything from March, all trend fields, all IDs.

## 5. Evaluation

**Client-holdout** (`GroupShuffleSplit` on `client_hash_id`, ~80/20 of clients) — the honest design for pages that share a client. Metrics: base rate, ROC AUC, average precision, precision@K, and lift over baseline — always next to the base rate. Error analysis: false positives tend to be high-volume pages that held steady; false negatives are low-volume pages whose drop is real but small.

## 6. Interpretation

The strongest features are volume, position, and content age — a "visible, aging, slipping" profile. Negative result worth stating: content type and intent carry little signal once volume and position are known. The model beats the baseline across review capacities, not just at one K.

## 7. Recommendation

A ranked queue: `refresh` (high risk + stale) > `refresh_review` (high risk) > `expand` (thin + high volume) > `review_ctr` (low CTR + high volume) > `monitor`. An editor starts at the top and works down with reason codes in hand. Confidence is explicit: this is decision support, not a causal claim.

## 8. Reproducibility

`hf auth login` → `python work/capstone_analysis_warehouse.py` → `python work/capstone_analysis_starter.py` → `python work/build_paper.py`. Seed 42, fixed build + windows. Env: Python 3.14, duckdb, scikit-learn, pandas, numpy, matplotlib (`requirements.txt`).

---

**Claims checklist:** observed / measured / directional / decision-support language everywhere · base rate reported next to every precision@K · no causal claim · no "predicted Google's algorithm" · no client-identifying detail · numbers match a fresh re-run.
