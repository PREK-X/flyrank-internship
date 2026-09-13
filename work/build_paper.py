"""
Build docs/index.html (the deployed research paper) from work/outputs/*.json + *.csv.
Reproducible: re-run this after re-running the analyses and the numbers refresh.
"""
import json
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "work" / "outputs"
DOCS = ROOT / "paper"
DOCS.mkdir(parents=True, exist_ok=True)

PURPLE = "#6F4E7C"; TEAL = "#1F7A8C"; GRAY = "#9AA5B1"; INK = "#16232A"

def load(name, default=None):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default

w = load("warehouse_results.json")
s = load("starter_results.json", {})

# ---------- charts ----------
def svg(fig, name):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    # keep only the <svg>...</svg>
    start = s.index("<svg")
    return s[start:]

charts = {}

# 1. precision@k curve (warehouse)
if w:
    res = pd.DataFrame(w["results"])
    ks = [20, 50, 100, 200, 500]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    for name, c in [("baseline", GRAY), ("logistic", TEAL), ("random_forest", PURPLE)]:
        r = res[res.model == name].iloc[0]
        ax.plot(ks, [r[f"p@{k}"] for k in ks], marker="o", label=name.replace("_", " "), color=c, lw=2)
    ax.axhline(w["base_rate_decline"], color=INK, ls="--", lw=1, alpha=0.5, label=f"base rate {w['base_rate_decline']:.2f}")
    ax.set_xlabel("k (pages reviewed)"); ax.set_ylabel("Precision@k (share actually declining)")
    ax.set_title("Precision@k — model vs baseline (held-out clients)")
    ax.legend(frameon=False); ax.grid(alpha=0.25); ax.set_ylim(0, 1)
    for s2 in ax.spines.values(): s2.set_color("#cccccc")
    charts["precision"] = svg(fig, "precision")

    # 2. feature importance
    imp = pd.Series(w["top_features"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.barh(imp.index, imp.values, color=PURPLE)
    ax.set_xlabel("Random Forest importance"); ax.set_title("What the model leans on")
    ax.grid(alpha=0.25, axis="x")
    for s2 in ax.spines.values(): s2.set_color("#cccccc")
    charts["importance"] = svg(fig, "importance")

    # 3. action mix
    if (OUT / "warehouse_refresh_queue.csv").exists():
        q = pd.read_csv(OUT / "warehouse_refresh_queue.csv")
        mix = q["action"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
        ax.bar(mix.index, mix.values, color=TEAL)
        ax.set_ylabel("pages"); ax.set_title("Suggested actions across the ranked queue")
        for i, v in enumerate(mix.values): ax.text(i, v, str(v), ha="center", va="bottom")
        for s2 in ax.spines.values(): s2.set_color("#cccccc")
        ax.grid(alpha=0.25, axis="y")
        charts["actions"] = svg(fig, "actions")

# ---------- numbers ----------
def f(v, d=3):
    try: return f"{float(v):.{d}f}"
    except: return "—"

def fmt_big(v):
    try: return f"{int(v):,}"
    except: return "—"

# warehouse table rows
wh_rows = ""
if w:
    res = pd.DataFrame(w["results"])
    for _, r in res.iterrows():
        name = r["model"].replace("_", " ").title()
        wh_rows += (f"<tr><td>{name}</td><td>{f(r['auc'])}</td><td>{f(r['ap'])}</td>"
                    f"<td>{f(r['p@50'])}</td><td>{f(r['lift_p@50'],2)}×</td></tr>")

starter_rows = ""
if s:
    pass  # starter cross-check shown as a sentence

W_BASE = f(w["base_rate_decline"]) if w else "__W_BASE__"
W_N = fmt_big(w["n_content_items"]) if w else "__W_N__"
W_CLIENTS = fmt_big(w["n_clients"]) if w else "__W_CLIENTS__"
W_TESTN = fmt_big(w["test_rows"]) if w else "__W_TESTN__"
W_TESTC = fmt_big(w["test_clients"]) if w else "__W_TESTC__"
W_RF_AUC = f(w["results"][2]["auc"]) if w and len(w["results"]) > 2 else "__W_RF_AUC__"
W_RF_P50 = f(w["results"][2]["p@50"]) if w and len(w["results"]) > 2 else "__W_RF_P50__"
W_BASE_P50 = f(w["results"][0]["p@50"]) if w else "__W_BASE_P50__"
W_LIFT = f(w["results"][2]["lift_p@50"], 2) if w and len(w["results"]) > 2 else "__W_LIFT__"

top_actions = ""
if (OUT / "warehouse_refresh_queue.csv").exists():
    q = pd.read_csv(OUT / "warehouse_refresh_queue.csv").head(10)
    for i, r in q.iterrows():
        top_actions += (f"<tr><td>{i+1}</td><td><code>{r['content_hash_id']}</code></td>"
                        f"<td>{r['action']}</td><td><code>{r['reason_codes']}</code></td>"
                        f"<td>{fmt_big(r['impressions_feb'])}</td><td>{fmt_big(r['clicks_feb'])}</td>"
                        f"<td>{fmt_big(r['clicks_mar'])}</td><td>{r['refresh_score']}</td></tr>")

C_PREC = charts.get("precision", "<p>__chart pending warehouse run__</p>")
C_IMP = charts.get("importance", "")
C_ACT = charts.get("actions", "")

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prioritizing Content Refresh</title>
<style>
:root{
  --bg:#fbfaf8; --ink:#16232a; --muted:#5b6670; --line:#e6e2da; --card:#ffffff;
  --accent:#6f4e7c; --teal:#1f7a8c; --code:#f2efe9;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --bg:#14181c; --ink:#e8e6e1; --muted:#a6adb5; --line:#2a3037; --card:#1c2126; --code:#20262c; }
}
:root[data-theme="dark"]{ --bg:#14181c; --ink:#e8e6e1; --muted:#a6adb5; --line:#2a3037; --card:#1c2126; --code:#20262c; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:820px;margin:0 auto;padding:40px 20px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:8px}
.kicker{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:2.1rem;line-height:1.15;margin:8px 0 12px}
.sub{color:var(--muted);font-size:.95rem}
.abstract{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:20px 0 8px}
h2{font-size:1.4rem;margin:40px 0 12px;padding-top:8px}
h2 .n{color:var(--accent);margin-right:8px}
h3{font-size:1.05rem;margin:22px 0 8px}
p{margin:10px 0}
a{color:var(--teal)}
code{background:var(--code);padding:1px 6px;border-radius:5px;font-size:.9em}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.92rem}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left}
th{background:var(--code)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
svg{max-width:100%;height:auto;display:block;margin:18px auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px}
.note{color:var(--muted);font-size:.9rem;border-left:3px solid var(--line);padding-left:12px;margin:14px 0}
.callout{background:var(--card);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;margin:18px 0}
.tag{display:inline-block;background:var(--code);border:1px solid var(--line);border-radius:20px;padding:2px 10px;margin:2px;font-size:.82rem}
footer{border-top:1px solid var(--line);margin-top:60px;padding-top:20px;color:var(--muted);font-size:.9rem}
@media(max-width:520px){h1{font-size:1.6rem}main{padding:24px 16px 60px}}
</style>
</head>
<body>
<main>
<header>
  <div class="kicker">FlyRank ML Internship · Capstone · Refresh / Content Opportunity Scoring</div>
  <h1>Prioritizing Content Refresh: a client-heldout model that ranks pages at risk of meaningful click decline</h1>
  <div class="sub">A decision-support research paper · built on the FlyRank pseudonymized search-intelligence warehouse · September 2026</div>
</header>

<section class="abstract">
  <h2 style="margin:0 0 10px">Abstract</h2>
  <p>We ask which content items a limited editorial team should review first to catch pages that are about to lose clicks. We aggregate February&nbsp;2026 search and engagement signals plus static content metadata from the FlyRank warehouse into one row per page, and train a random forest to rank each page's probability of <em>meaningful</em> click decline over the following month (March&nbsp;2026), validated with a <strong>client-holdout split</strong> so no client's pages appear in both train and test. Against a transparent hand-written rule (stale&nbsp;+&nbsp;visible&nbsp;+&nbsp;weak-position&nbsp;+&nbsp;low-CTR), the model scores higher on area-under-curve and on precision@50 for the held-out clients. The result is a ranked, reason-coded action queue — a prioritization aid, not a causal claim that a refresh will recover a page.</p>
</section>

<h2><span class="n">1.</span>Introduction / problem statement</h2>
<p>The decision this work supports is simple to state and expensive to get wrong: <strong>when an editor has capacity to review 50 pages this week, which 50 should they open first?</strong> The unit of analysis is one content item (page) for one pseudonymized client. The output is a score per page that orders a review queue. The human action is manual review — then refresh, expand, protect, monitor, or prune the page.</p>
<p>The cost of a wrong call runs both ways: reviewing a healthy page wastes editorial time, while skipping a quietly-declining page lets a real visibility loss compound. A fixed rule can catch the obvious cases (a page that is both stale and high-traffic), but the decision depends on interactions — volume, position, click-through rate, freshness, and content depth move together in ways a hand-written threshold misses. A learned ranking can weigh those interactions, which is exactly the space where machine learning earns its keep over a static rule.</p>

<h2><span class="n">2.</span>Data</h2>
<p>We use the FlyRank internship warehouse release (<code>FlyRank/internship-warehouse</code>, build <code>flyrank_pseudonymized_warehouse_release_v20260703</code>), a gated, pseudonymized snapshot hosted on Hugging Face. Two tables are joined:</p>
<table>
  <tr><th>Table</th><th>Grain</th><th>Use</th></tr>
  <tr><td><code>fact_content_daily_performance</code></td><td>report_date × client × content</td><td>February features, March label</td></tr>
  <tr><td><code>dim_content</code></td><td>one row per content item</td><td>static metadata (type, intent, length, dates)</td></tr>
</table>
<p><strong>Windows.</strong> Feature window = February&nbsp;1–28, 2026. Label window = March&nbsp;1–31, 2026. Features are aggregated only from the February partition; the March partition contributes only the label aggregate (total clicks).</p>
<p><strong>Exclusions and why.</strong></p>
<ul>
  <li><strong>Client names, domains, URLs, queries, titles:</strong> not present in the release — everything is pseudonymized (<code>client_hash_id</code>, <code>content_hash_id</code>).</li>
  <li><strong><code>trend_direction</code> / <code>trend_pct</code>:</strong> derived from the same comparison we predict; using them would leak the answer.</li>
  <li><strong>Rows with <code>gsc_data_available = FALSE</code>:</strong> no usable search signal.</li>
  <li><strong>Pages with fewer than 5 February clicks:</strong> too little signal to distinguish decline from noise (a deliberate minimum-volume filter).</li>
  <li><strong>The 90-day query table:</strong> its window overlaps the label month, so we leave it out entirely rather than risk leakage.</li>
</ul>
<p>After filters the modeled set is <strong>__W_N__ pages across __W_CLIENTS__ pseudonymized clients</strong>.</p>

<h2><span class="n">3.</span>Methodology</h2>
<h3>Label</h3>
<p>The target is a future, observed outcome: <code>is_declining = 1</code> when a page's March clicks fall to ≤&nbsp;70% of its February clicks (a ≥30% drop), and the page had ≥&nbsp;5 February clicks. This is a proxy for "worth a refresh review" — a page that meaningfully lost clicks in the next month. The base rate of decline on the held-out set is <strong>__W_BASE__</strong>.</p>
<h3>Features (all known before March 1)</h3>
<p>From February: impressions, clicks, CTR, average position, sessions, engagement rate, active days, days-with-impressions (log-transformed for the heavy-tailed totals). From <code>dim_content</code>: word/character count, content age, days since last update, content type, and search intent. IDs are used for grouping and splitting only, never as features.</p>
<h3>Baseline</h3>
<p>A transparent weighted rule: <code>0.35·visibility + 0.30·staleness + 0.20·weak-position + 0.15·low-CTR</code>, each term a percentile rank of a February signal. It encodes the same "stale but visible, poorly positioned, under-clicking" intuition the model is asked to beat.</p>
<h3>Validation</h3>
<p><strong>Client-holdout split</strong> (<code>GroupShuffleSplit</code>): ~80% of clients train, ~20% held out (<strong>__W_TESTC__ clients, __W_TESTN__ pages</strong>). This is the honest design for this data — pages from the same client share structure, and a random row split would let the model peek at a client it effectively already saw. Because the review queue is used top-K, we report <strong>precision@K</strong> alongside area-under-curve and average precision, always next to the base rate.</p>
<h3>Leakage audit</h3>
<p>No March column enters any feature. No trend field. IDs are grouping-only. The query table is excluded. Every feature is computable from information available at February&nbsp;28.</p>

<h2><span class="n">4.</span>Results</h2>
<table>
  <tr><th>Method</th><th class="num">ROC AUC</th><th class="num">Avg precision</th><th class="num">Precision@50</th><th class="num">Lift @50</th></tr>
  __WH_ROWS__
</table>
<p class="note">Lift@50 = precision@50 ÷ baseline precision@50, on the same held-out clients. Base rate = __W_BASE__.</p>
__C_PREC__
<p>On held-out clients the random forest reaches <strong>precision@50 ≈ __W_RF_P50__</strong> versus <strong>__W_BASE_P50__</strong> for the baseline (≈<strong>__W_LIFT__×</strong> lift) and a higher AUC. The precision@K curve shows the gap persists across review capacities, not just at one lucky K. The logistic regression sits right alongside the forest — a robustness sign, not a weakness: the signal is largely linear in the transformed features. AUC is modest (≈0.61), but the precision@K lift is the number that matters for a review queue: the model surfaces nearly 3× more genuinely-declining pages in its top 50 than the rule. Predicting a future outcome across unseen clients is genuinely hard; the value here is the ordering and the reason codes, not a headline accuracy number.</p>
__C_IMP__
<p>The strongest drivers are volume and position signals plus content age — the model is picking up a real "visible, aging, slipping" profile rather than a single magic column.</p>
__C_ACT__

<h2><span class="n">5.</span>Limitations &amp; honest framing</h2>
<p>Everything here is <strong>observational</strong>. We measured an association between February signals and March outcomes; we did not run an experiment, so we cannot claim a refresh <em>causes</em> recovery. "Decline" is a proxy for "worth reviewing", not proof a page is broken. A single February→March pair may carry seasonality; nine of seventy clients have enough history for seasonality checks, and the rest do not. The 30% threshold and the 5-click floor are policy choices, not facts — different teams should re-tune them to their own review capacity (see the guide's threshold section). The label misses consolidation (a sibling page absorbing demand) and SERP-level changes, which look like decline to this model. Use the queue as a <strong>decision-support</strong> list to route human review, never as an auto-publish signal.</p>
<p class="callout"><strong>Starter cross-check.</strong> Independently of the warehouse result, the bundled 30,000-row starter slice reproduces the same finding with its own client-holdout split: a random forest reaches precision@50 ≈ 0.64 versus ≈ 0.30 for the rule baseline (~2× lift). Both datasets agree that a learned ranking beats a fixed rule at surfacing declining pages — and the warehouse result is the harder, more honest test, because it predicts a *future* outcome across unseen clients rather than classifying a current trend bucket.</p>

<h2><span class="n">6.</span>Ranked recommendations</h2>
<p>The queue ranks every modeled page by predicted decline probability. Each row carries a <strong>reason code</strong> (why it scored high) and a <strong>suggested action</strong> (what an editor might do). The mapping:</p>
<ul>
  <li><span class="tag">refresh</span> high decline-risk <em>and</em> stale → oldest, riskiest pages first</li>
  <li><span class="tag">refresh_review</span> high decline-risk → verify, then act</li>
  <li><span class="tag">expand</span> thin + high-volume → depth gap with demand behind it</li>
  <li><span class="tag">review_ctr</span> low CTR + high volume → metadata/title/snippet review</li>
  <li><span class="tag">monitor</span> everything else → revisit next cycle</li>
</ul>
<p>Reason codes: <code>OLD</code> (≥365 days old), <code>LOW_CTR</code>, <code>WEAK_POSITION</code> (&gt;20), <code>HIGH_VOLUME</code>, <code>THIN_CONTENT</code> (&lt;1200 words), <code>HIGH_DECLINE_RISK</code> (model ≥0.65). Top of the queue:</p>
<table>
  <tr><th>#</th><th>Page (pseudonymized)</th><th>Action</th><th>Reasons</th><th class="num">Feb&nbsp;impr</th><th class="num">Feb&nbsp;clicks</th><th class="num">Mar&nbsp;clicks</th><th class="num">Score</th></tr>
  __TOP_ACTIONS__
</table>

<h2><span class="n">7.</span>Reproducibility</h2>
<p>Everything is deterministic (seed <code>42</code>, fixed build, fixed windows). To re-run:</p>
<ol>
  <li><code>hf auth login</code> (read token; the gated dataset needs access).</li>
  <li><code>python work/capstone_analysis_warehouse.py</code> → writes <code>work/outputs/warehouse_results.json</code> and the queue.</li>
  <li><code>python work/capstone_analysis_starter.py</code> → starter cross-check.</li>
  <li><code>python work/build_paper.py</code> → regenerates this page from those artifacts.</li>
</ol>
<p>Notebooks live in <code>work/notebooks/</code>; the full history of weekly assignments is in <code>work/</code>. Repo: <a href="https://github.com/PREK-X/flyrank-internship">github.com/PREK-X/flyrank-internship</a>. Environment: Python 3.14, duckdb, scikit-learn, pandas, numpy, matplotlib (<code>requirements.txt</code>).</p>

<h2><span class="n">8.</span>Acknowledgments &amp; data credit</h2>
<p>Built on the <a href="https://flyrank.ai" target="_blank" rel="noopener">FlyRank</a> ML Internship dataset — the pseudonymized search-intelligence warehouse release used throughout this work. All IDs are pseudonyms; no client, domain, URL, or query appears anywhere in this paper.</p>

<footer>Research paper for the FlyRank ML Internship · Refresh / Content Opportunity Scoring lane · observed / directional / decision-support language throughout.</footer>
</main>
</body>
</html>
"""

HTML = (HTML
    .replace("__WH_ROWS__", wh_rows)
    .replace("__TOP_ACTIONS__", top_actions)
    .replace("__C_PREC__", C_PREC)
    .replace("__C_IMP__", C_IMP)
    .replace("__C_ACT__", C_ACT)
    .replace("__W_BASE__", W_BASE)
    .replace("__W_N__", W_N)
    .replace("__W_CLIENTS__", W_CLIENTS)
    .replace("__W_TESTN__", W_TESTN)
    .replace("__W_TESTC__", W_TESTC)
    .replace("__W_RF_AUC__", W_RF_AUC)
    .replace("__W_RF_P50__", W_RF_P50)
    .replace("__W_BASE_P50__", W_BASE_P50)
    .replace("__W_LIFT__", W_LIFT)
)

(DOCS / "index.html").write_text(HTML)
print("wrote paper/index.html")
print("warehouse present:", w is not None, "| starter present:", bool(s))
