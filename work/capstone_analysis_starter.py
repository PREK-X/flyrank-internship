"""
Refresh / Content Opportunity Scoring — starter-dataset analysis (runnable locally, no HF).
Client-holdout validation. Baseline rule vs LogisticRegression vs RandomForest.
Reproduces the reference pipeline numbers and adds lift + precision@k curve.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

df = pd.read_csv("data/raw/content_refresh_anonymized.csv")

# ---- Label (proxy: current-window decline; matches reference pipeline) ----
df["is_declining"] = (df["trend_direction"] == "down").astype(int)

# ---- Features (numeric: log-transformed heavy tails + rates; no trend/IDs) ----
num_feats = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    "days_with_impressions", "days_with_sessions",
    "content_age_days", "days_since_last_update",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
]
cat_feats = ["competition_level", "content_type", "main_intent",
             "age_tier", "freshness_tier", "word_count_tier", "impression_tier", "position_tier"]

# log1p transforms (data-dictionary gotcha: rate cols are x100 percentages, keep as-is)
for c in ["impressions_90d", "clicks_90d", "sessions_90d", "ai_sessions_90d"]:
    df[f"log_{c}"] = np.log1p(df[c])

# missingness flags: blank word_count encodes content_type — flag, don't fillna(0)
df["has_word_count"] = df["word_count"].notna().astype(int)
df["has_keyword_data"] = df["search_volume"].notna().astype(int)

X_num = df[num_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
enc = OneHotEncoder(handle_unknown="ignore")
X_cat = enc.fit_transform(df[cat_feats].fillna("unknown"))
X = hstack([X_num, X_cat]).tocsr()
y = df["is_declining"].values
groups = df["client_id"].values

# ---- Split: client-holdout (train on ~80% of clients, test on held-out clients) ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))
y_tr, y_te = y[tr_idx], y[te_idx]
base_rate = y_te.mean()

# ---- Baseline: transparent hand rule (visibility + freshness + position + depth) ----
def baseline_score(d):
    vis   = d["impressions_90d"].rank(pct=True).fillna(0)
    fresh = d["days_since_last_update"].rank(pct=True).fillna(0)     # staler = higher
    pos   = (1 - d["avg_position"].rank(pct=True).fillna(0))          # weak position = opportunity
    depth = (1 - d["word_count"].rank(pct=True).fillna(0))            # thin = depth gap
    return 0.40*vis + 0.30*fresh + 0.25*pos + 0.05*depth

base_all = baseline_score(df).values
base_te = base_all[te_idx]

# ---- Models ----
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1).fit(X[tr_idx], y_tr)
lr = LogisticRegression(max_iter=2000, random_state=42).fit(X[tr_idx], y_tr)

def prec_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1][:k]
    return y_true[order].mean()

ks = [20, 50, 100, 200, 500]
rows = []
for name, scores in [("baseline", base_te), ("logistic", lr.predict_proba(X[te_idx])[:,1]),
                     ("random_forest", rf.predict_proba(X[te_idx])[:,1])]:
    rows.append({
        "model": name,
        "auc": roc_auc_score(y_te, scores),
        "ap": average_precision_score(y_te, scores),
        **{f"p@{k}": prec_at_k(y_te, scores, k) for k in ks},
    })
res = pd.DataFrame(rows)
res["lift_p@50_over_baseline"] = res["p@50"] / res.loc[res.model=="baseline","p@50"].values[0]

print("=== client-holdout results (held-out clients only) ===")
print(f"test rows: {len(y_te)}, held-out clients: {len(np.unique(groups[te_idx]))}")
print(f"base rate (declining): {base_rate:.3f}")
print(res.round(3).to_string(index=False))
res.to_csv("work/outputs/starter_results.csv", index=False)

# ---- Feature importance (RF) ----
imp = pd.Series(rf.feature_importances_, index=list(num_feats) + list(enc.get_feature_names_out(cat_feats)))
imp.sort_values(ascending=False).head(15).to_csv("work/outputs/starter_feature_importance.csv")
print("\n=== top features ===")
print(imp.sort_values(ascending=False).head(10).round(3).to_string())

# ---- Ranked recommendations (full data, RF probability) ----
rf_proba_all = rf.predict_proba(X)[:, 1]
out = df[["content_id", "client_id", "impressions_90d", "clicks_90d", "avg_position",
          "ctr", "days_since_last_update", "word_count", "trend_direction"]].copy()
out["refresh_score"] = (100 * rf_proba_all).round(1)
out = out.sort_values("refresh_score", ascending=False)
out.to_csv("work/outputs/starter_refresh_queue.csv", index=False)
print("\n=== top 10 queue preview ===")
print(out.head(10)[["content_id","refresh_score","impressions_90d","avg_position","trend_direction"]].to_string(index=False))
print("\nDONE. artifacts in work/outputs/")
