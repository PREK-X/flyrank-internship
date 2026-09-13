# Capstone Runbook — what's left to finish

Your capstone is ~90% built. This file is the exact list of what remains — all of it needs you (credentials + one deploy), not more code.

## What's already done (this session)

| File | What it is | Status |
|---|---|---|
| `paper/index.html` | The deployed research paper (9 sections) | Written, numbers auto-fill after the warehouse run |
| `work/capstone_analysis_warehouse.py` | Primary model: Feb 2026 → March 2026, client-holdout, RF vs baseline | Written, **needs HF token to run** |
| `work/capstone_analysis_starter.py` | Starter-slice cross-check (30k rows, local) | **Already ran** → real numbers |
| `work/explore_warehouse.py` | Confirms schemas before the big run | Written |
| `work/build_paper.py` | Regenerates `paper/index.html` from the result JSONs | Written, ran once |
| `work/notebooks/capstone.ipynb` | The capstone notebook | Written |
| `work/capstone_report.md` | The 8-section report | Written |
| `.github/workflows/pages.yml` | Deploys `docs/` to GitHub Pages on push | Written |
| `submission/paper_url.txt` | One-line URL of the deployed paper | Placeholder — fill after deploy |

## Step 1 — Authenticate to Hugging Face (one-time, ~2 min)

1. If you haven't: request access at https://huggingface.co/datasets/FlyRank/internship-warehouse (accept the data-use terms — instant).
2. Create a READ token at https://huggingface.co/settings/tokens.
3. In this terminal run:
   ```
   hf auth login
   ```
   and paste the token. (Or: `mkdir -p ~/.cache/huggingface && echo -n "hf_..." > ~/.cache/huggingface/token`)

## Step 2 — Run the warehouse analysis (the primary result)

```
cd "/home/prek-x/Documents/flyrank internship/flyrank-internship"
./venv/bin/python work/explore_warehouse.py            # confirm schemas
./venv/bin/python work/capstone_analysis_warehouse.py   # full model + queue
./venv/bin/python work/build_paper.py                  # fill real numbers into paper/index.html
```

If `explore_warehouse.py` shows different `dim_content` column names than my guesses, tell me and I'll fix `capstone_analysis_warehouse.py` in one edit.

## Step 3 — Deploy to GitHub Pages (one-time, ~3 min)

1. `git add -A && git commit -m "Capstone: paper, model, notebook, report" && git push`
2. In the repo on GitHub → **Settings → Pages → Source = GitHub Actions**.
3. The `pages.yml` workflow runs on push and publishes `paper/`. Your URL is:
   ```
   https://prek-x.github.io/flyrank-internship/
   ```
4. Put that exact URL (one line, nothing else) into `submission/paper_url.txt`, commit and push it.

## Step 4 — Submit

On the capstone card in your portal, submit your repo URL only: `https://github.com/PREK-X/flyrank-internship`. The paper is found through `submission/paper_url.txt`.

## The two things only you can do

1. **HF token** (Step 1–2) — I have no credentials here, so I can't run the warehouse myself.
2. **Push + enable Pages** (Step 3) — publishing is your GitHub account's action.

Everything else is built. Do Step 1 and tell me — I'll finish the warehouse run and finalize the numbers.
