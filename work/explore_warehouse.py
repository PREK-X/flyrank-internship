"""One-shot warehouse exploration: confirm token, table schemas, row counts, date windows.
Run this first (after `hf auth login`) to confirm column names before the full analysis."""
import duckdb
from pathlib import Path

TOK = Path.home() / ".cache" / "huggingface" / "token"
if not TOK.exists():
    raise SystemExit("No token at ~/.cache/huggingface/token — run `hf auth login` first")
token = TOK.read_text().strip()
if not token.startswith("hf_"):
    token = token.splitlines()[0] if token else ""

con = duckdb.connect()
con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")

REL = "hf://datasets/FlyRank/internship-warehouse"
FACT = f"{REL}/fact_content_daily_performance"

print("=== dim_clients schema ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{REL}/dim_clients.parquet')").df()[["column_name","column_type"]].to_string(index=False))

print("\n=== dim_content schema ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{REL}/dim_content.parquet')").df()[["column_name","column_type"]].to_string(index=False))

print("\n=== fact daily schema (Feb sample) ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{FACT}/month=2026-02/*.parquet') LIMIT 1").df()[["column_name","column_type"]].to_string(index=False))

print("\n=== row counts Feb vs Mar (gsc available) ===")
for m in ["2026-02","2026-03"]:
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{FACT}/month={m}/*.parquet') WHERE gsc_data_available IS TRUE").fetchone()[0]
    print(f"  {m}: {n:,} rows with gsc_data_available")

print("\n=== dim_content sample (5 rows) ===")
print(con.execute(f"SELECT * FROM read_parquet('{REL}/dim_content.parquet') LIMIT 5").df().to_string())
