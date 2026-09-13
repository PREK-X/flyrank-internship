"""
Refresh / Content Opportunity Scoring — warehouse future-window analysis (PRIMARY capstone result).

Design (leakage-safe):
  feature window = February 2026 (observed signals + static content metadata)
  label window   = March 2026 (future outcome)
  label          = meaningful click decline: clicks_feb >= 5 AND clicks_mar <= 0.7 * clicks_feb
  validation     = client-holdout (train on ~80% of clients, test on held-out clients)
  metrics        = base rate, ROC AUC, avg precision, precision@K, lift over baseline

Run after `hf auth login`. Writes everything to work/outputs/.
"""
import json
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(".").resolve()
OUT = ROOT / "work" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- auth ----
TOK = Path.home() / ".cache" / "huggingface" / "token"
if not TOK.exists():
    raise SystemExit("No token at ~/.cache/huggingface/token — run `hf auth login` first")
token = TOK.read_text().strip().splitlines()[0]

con = duckdb.connect()
con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")

# cache the two big downloads so re-runs skip the network
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

def cached_sql(name, sql):
    cp = CACHE / f"{name}.csv"
    if cp.exists():
        return pd.read_csv(cp)
    df = con.execute(sql).df()
    df.to_csv(cp, index=False)
    return df

REL = "hf://datasets/FlyRank/internship-warehouse"
FACT = f"{REL}/fact_content_daily_performance"
FEB = f"{FACT}/month=2026-02/*.parquet"
MAR = f"{FACT}/month=2026-03/*.parquet"
DIMC = f"{REL}/dim_content.parquet"

# ---------------------------------------------------------------------------
# 1. Feature frame (February, aggregated per content item) + dim_content join
# ---------------------------------------------------------------------------
features = cached_sql("features_feb", f"""
WITH feb AS (
    SELECT client_hash_id, content_hash_id,
           SUM(gsc_impressions)              AS impressions_feb,
           SUM(gsc_clicks)                   AS clicks_feb,
           AVG(gsc_avg_position)             AS avg_position_feb,
           SUM(ga4_sessions)                 AS sessions_feb,
           SUM(ga4_engaged_sessions)         AS engaged_sessions_feb,
           SUM(sessions_ai)                  AS ai_sessions_feb,
           COUNT(DISTINCT report_date)       AS active_days_feb,
           COUNT(DISTINCT CASE WHEN gsc_impressions > 0 THEN report_date END) AS days_with_impressions_feb
    FROM read_parquet('{FEB}')
    WHERE gsc_data_available IS TRUE
    GROUP BY 1, 2
)
SELECT f.*,
       c.content_type, c.main_intent, c.word_count, c.char_count,
       c.search_volume, c.competition, c.backlinks,
       c.content_created_date, c.content_updated_date,
       c.is_published, c.is_deleted
FROM feb f
LEFT JOIN read_parquet('{DIMC}') c USING (content_hash_id)
WHERE c.is_published IS TRUE AND c.is_deleted IS FALSE
""")

# engineered Feb features (leakage-safe: all known at Feb 28)
features["ctr_feb"] = 100.0 * features["clicks_feb"] / features["impressions_feb"].replace(0, np.nan)
features["engagement_rate_feb"] = 100.0 * features["engaged_sessions_feb"] / features["sessions_feb"].replace(0, np.nan)
features["log_impressions_feb"] = np.log1p(features["impressions_feb"])
features["log_clicks_feb"] = np.log1p(features["clicks_feb"])
features["log_sessions_feb"] = np.log1p(features["sessions_feb"])
features["log_ai_sessions_feb"] = np.log1p(features["ai_sessions_feb"])
features["log_search_volume"] = np.log1p(features["search_volume"].fillna(0))
features["log_backlinks"] = np.log1p(features["backlinks"].fillna(0))
features["content_age_days_feb"] = (pd.Timestamp("2026-02-28") - pd.to_datetime(features["content_created_date"], errors="coerce")).dt.days
features["has_word_count"] = features["word_count"].notna().astype(int)

# ---------------------------------------------------------------------------
# 2. Label (March = future outcome)
# ---------------------------------------------------------------------------
mar = cached_sql("mar", f"""
    SELECT client_hash_id, content_hash_id, SUM(gsc_clicks) AS clicks_mar
    FROM read_parquet('{MAR}')
    WHERE gsc_data_available IS TRUE
    GROUP BY 1, 2
""")

data = features.merge(mar, on=["client_hash_id", "content_hash_id"], how="left")
data["clicks_mar"] = data["clicks_mar"].fillna(0)

# min-volume eligibility (can't detect meaningful decline without enough Feb signal)
data = data[data["clicks_feb"] >= 5].copy()

# meaningful decline: March clicks at least 30% below February
data["is_declining"] = (data["clicks_mar"] <= 0.7 * data["clicks_feb"]).astype(int)

# ---------------------------------------------------------------------------
# 3. Feature matrix
# ---------------------------------------------------------------------------
num_feats = [
    "impressions_feb", "log_impressions_feb", "clicks_feb", "log_clicks_feb",
    "ctr_feb", "avg_position_feb", "sessions_feb", "log_sessions_feb",
    "engagement_rate_feb", "active_days_feb", "days_with_impressions_feb",
    "log_ai_sessions_feb", "log_search_volume", "log_backlinks",
    "word_count", "char_count", "content_age_days_feb",
    "has_word_count",
]
cat_feats = ["content_type", "main_intent"]

X_num = data[num_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
dummies = pd.get_dummies(data[cat_feats].fillna("unknown"), prefix=cat_feats, dtype=float)
X_cat = dummies.values
X = np.hstack([X_num, X_cat])
y = data["is_declining"].values
groups = data["client_hash_id"].values
feat_names = num_feats + list(dummies.columns)

# ---------------------------------------------------------------------------
# 4. Client-holdout split
# ---------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))
y_tr, y_te = y[tr_idx], y[te_idx]
base_rate = y_te.mean()

# ---------------------------------------------------------------------------
# 5. Baseline (transparent rule: stale + visible + weak position + low CTR)
# ---------------------------------------------------------------------------
def baseline_score(d):
    vis    = d["impressions_feb"].rank(pct=True).fillna(0)
    stale  = d["content_age_days_feb"].rank(pct=True).fillna(0)
    weakpos = d["avg_position_feb"].clip(lower=0, upper=50).rank(pct=True).fillna(0)
    lowctr = (100 - d["ctr_feb"].clip(upper=10)).rank(pct=True).fillna(0)
    return (0.35*vis + 0.30*stale + 0.20*weakpos + 0.15*lowctr).clip(0, 1)

base_all = baseline_score(data).values
base_te = base_all[te_idx]

# ---------------------------------------------------------------------------
# 6. Models + metrics
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=25,
                            class_weight="balanced_subsample", n_jobs=-1, random_state=42)
rf.fit(X[tr_idx], y_tr)

sc = StandardScaler().fit(X[tr_idx])
lr = LogisticRegression(max_iter=3000, random_state=42)
lr.fit(sc.transform(X[tr_idx]), y_tr)

def prec_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1][:k]
    return y_true[order].mean()

ks = [20, 50, 100, 200, 500]
rows = []
for name, scores in [("baseline", base_te),
                     ("logistic", lr.predict_proba(sc.transform(X[te_idx]))[:, 1]),
                     ("random_forest", rf.predict_proba(X[te_idx])[:, 1])]:
    rows.append({
        "model": name,
        "auc": roc_auc_score(y_te, scores),
        "ap": average_precision_score(y_te, scores),
        **{f"p@{k}": prec_at_k(y_te, scores, k) for k in ks},
    })
res = pd.DataFrame(rows)
base_p50 = res.loc[res.model == "baseline", "p@50"].values[0]
res["lift_p@50"] = res["p@50"] / base_p50

# ---------------------------------------------------------------------------
# 7. Feature importance + error analysis
# ---------------------------------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

# error analysis on held-out clients
rf_pred = (rf.predict_proba(X[te_idx])[:, 1] >= 0.5).astype(int)
fp = (rf_pred == 1) & (y_te == 0)
fn = (rf_pred == 0) & (y_te == 1)
fp_imp = data["impressions_feb"].values[te_idx][fp].mean()
fn_imp = data["impressions_feb"].values[te_idx][fn].mean()

# ---------------------------------------------------------------------------
# 8. Ranked queue + reason codes + actions
# ---------------------------------------------------------------------------
rf_proba = rf.predict_proba(X)[:, 1]
data["decline_probability"] = rf_proba
data["refresh_score"] = (100 * rf_proba).round(1)

for _c in ["avg_position_feb", "ctr_feb",
            "word_count", "char_count", "content_age_days_feb"]:
    data[_c] = data[_c].fillna(0)
data["content_type"] = data["content_type"].fillna("unknown")
data["main_intent"] = data["main_intent"].fillna("unknown")

ctr_med = data["ctr_feb"].median()
imp_p75 = data["impressions_feb"].quantile(0.75)

def reason_codes(r):
    rs = []
    if r["content_age_days_feb"] >= 365: rs.append("OLD")
    if r["ctr_feb"] < ctr_med:            rs.append("LOW_CTR")
    if r["avg_position_feb"] > 20:        rs.append("WEAK_POSITION")
    if r["impressions_feb"] >= imp_p75:   rs.append("HIGH_VOLUME")
    if r["word_count"] > 0 and r["word_count"] < 1200: rs.append("THIN_CONTENT")
    if r["decline_probability"] >= 0.65:  rs.append("HIGH_DECLINE_RISK")
    return "|".join(rs) if rs else "MONITOR"

def action(r):
    rs = reason_codes(r)
    if "OLD" in rs and "HIGH_VOLUME" in rs: return "refresh"
    if "THIN_CONTENT" in rs and "HIGH_VOLUME" in rs: return "expand"
    if "HIGH_DECLINE_RISK" in rs: return "refresh_review"
    if "LOW_CTR" in rs and "HIGH_VOLUME" in rs: return "review_ctr"
    if r["impressions_feb"] < 20 and r["clicks_feb"] < 3: return "prune_or_monitor"
    return "monitor"

data["reason_codes"] = data.apply(reason_codes, axis=1)
data["action"] = data.apply(action, axis=1)

queue = data.sort_values("refresh_score", ascending=False)[
    ["content_hash_id", "client_hash_id", "refresh_score", "decline_probability",
     "action", "reason_codes", "impressions_feb", "clicks_feb", "clicks_mar",
     "ctr_feb", "avg_position_feb", "content_age_days_feb",
     "word_count", "content_type", "main_intent"]
].copy()
queue["decline_probability"] = queue["decline_probability"].round(3)
queue["ctr_feb"] = queue["ctr_feb"].round(2)
queue["avg_position_feb"] = queue["avg_position_feb"].round(1)

# ---------------------------------------------------------------------------
# 9. Export
# ---------------------------------------------------------------------------
res.to_csv(OUT / "warehouse_results.csv", index=False)
imp.head(20).to_csv(OUT / "warehouse_feature_importance.csv")
queue.to_csv(OUT / "warehouse_refresh_queue.csv", index=False)

summary = {
    "n_content_items": int(len(data)),
    "n_clients": int(data["client_hash_id"].nunique()),
    "test_rows": int(len(y_te)),
    "test_clients": int(len(np.unique(groups[te_idx]))),
    "base_rate_decline": float(base_rate),
    "label_definition": "clicks_feb >= 5 AND clicks_mar <= 0.7 * clicks_feb",
    "results": res.round(4).to_dict(orient="records"),
    "top_features": {k: round(float(v), 4) for k, v in imp.head(12).items()},
    "fp_mean_impressions": float(fp_imp),
    "fn_mean_impressions": float(fn_imp),
    "action_mix": data["action"].value_counts().to_dict(),
}
(OUT / "warehouse_results.json").write_text(json.dumps(summary, indent=2))

print("=== warehouse future-window results (client-holdout) ===")
print(f"content items: {len(data):,} | clients: {data['client_hash_id'].nunique()}")
print(f"test rows: {len(y_te):,} | held-out clients: {len(np.unique(groups[te_idx]))}")
print(f"base rate (meaningful decline): {base_rate:.3f}")
print(res.round(3).to_string(index=False))
print("\n=== top features ===")
print(imp.head(10).round(3).to_string())
print("\n=== action mix ===")
print(data["action"].value_counts().to_string())
print("\n=== error analysis (held-out) ===")
print(f"false-positive mean Feb impressions: {fp_imp:.0f} | false-negative mean Feb impressions: {fn_imp:.0f}")
print("\n=== top 15 queue ===")
print(queue.head(15)[["content_hash_id","refresh_score","action","reason_codes","impressions_feb","clicks_feb","clicks_mar"]].to_string(index=False))
print("\nDONE. artifacts in work/outputs/")
