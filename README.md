# HumAID Tweet Classification (Zero-shot with OpenAI Batch)

[![Open results dashboard](https://img.shields.io/badge/Results-Dashboard-0a7)](https://htmlpreview.github.io/?https://github.com/anhtranst/humaid/blob/main/results/index.html)

> 🔗 **Live Results Dashboard:**  
> https://htmlpreview.github.io/?https://github.com/anhtranst/humaid/blob/main/results/index.html

Zero-shot tweet classification for humanitarian response categories (HumAID-style labels) using OpenAI's **Chat Completions + Structured Outputs** and the **Batch API**.  
Outputs include predictions and analysis artifacts (confusion matrices, per-class metrics, mistakes), plus a **curated results dashboard**.

## Labels (canonical order)

```
caution_and_advice
sympathy_and_support
requests_or_urgent_needs
displaced_people_and_evacuations
injured_or_dead_people
missing_or_found_people
infrastructure_and_utility_damage
rescue_volunteering_or_donation_effort
other_relevant_information
not_humanitarian
```

This order is enforced in the code (enum for Structured Outputs) and used consistently in plots and reports.

---

## Quick Start

### 1) Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`, `matplotlib`, `requests`, `python-dotenv`
- Optional (recommended): `tiktoken` for accurate token estimates
- An OpenAI API key (models: `gpt-4o-mini` recommended)

```bash
pip install -r requirements.txt
# or:
pip install pandas numpy matplotlib requests python-dotenv tiktoken
```

### 2) Configure your API keys (do NOT commit them)

Create a `.env` file in the repo root. You can keep a primary and an alternate key:

```
OPENAI_API_KEY_1=sk-...  # e.g., Tier-1 key with 2M batch cap
OPENAI_API_KEY_2=sk-...  # e.g., higher-capacity key
# Optional: set a default
OPENAI_API_KEY=${OPENAI_API_KEY_1}
```

`.gitignore` already ignores `.env`. If a key was ever committed, rotate it and purge from history.

### 3) Project layout

```
Dataset/
  HumAID/<event>/<event>_<split>.tsv     # Input TSVs

humaidclf/
  __init__.py        # Exposes the package API (incl. run_experiment, resume_experiment)
  io.py              # load_tsv, plan_run_dirs
  prompts.py         # LABELS (ordered as above), SYSTEM_PROMPT, make_user_message
  batch.py           # sync_test_sample, build_requests_jsonl_S, batch helpers, parser, API key switchers
  budget.py          # Token budgeting (dataset token estimates, sharding, gating by cap)
  eval.py            # Metrics + analysis & plots (fixed canonical label order)
  report.py          # NEW: curated results index (HTML) with zoomable charts
  runner.py          # High-level orchestration: run_experiment(), resume_experiment()

rules/               # project-local rule variants (outside the package for fast iteration)
  __init__.py        # exports RULES_BASELINE, RULES_1, RULES_REGISTRY, get_rule
  humaid_rules.py    # definitions of RULES_BASELINE and RULES_1 (edit here as you iterate)

runs/
  <event>/<split>/<model>/<timestamp>-<tag>/
    requests.jsonl
    outputs.jsonl
    predictions.csv
    analysis/                      # per-run analysis lives here
      mistakes.csv
      charts/
        confusion_matrix_counts.png
        confusion_matrix_row_normalized.png
        confusion_matrix.csv
        confusion_matrix_row_normalized.csv
        per_class_f1.png
        per_class_error_rate.png
        top_confusions.png
        per_class_metrics.csv
        summary.json
  _indexes/
    token_budget_*.csv             # saved token budgeting indices
    runs_*.csv                     # indices of completed runs

results/            # NEW: curated, human-picked best results (one or more per event/split/model)
  <event>/<split>/<model>/<run_name>/
    predictions.csv
    analysis/
      charts/ (same artifacts as runs/*/analysis/charts/)
      summary.json
      mistakes.csv
  index.html        # built by report.py (summary table + detailed cards)

00_build_results_index.ipynb   # NEW: small notebook to rebuild results/index.html
```

We intentionally store **analysis** inside each run folder so repeated analyses don't mix across runs. The `results/` tree is **manual and curated**: copy only your favorite run(s) there for presentation.

---

## How to Use

> You can use either the **one-liner runner** or the **step-by-step** workflow.

### Option 1 — One-liner runner (recommended)

```python
from dotenv import load_dotenv; load_dotenv()
from rules import RULES_BASELINE  # or RULES_1
from humaidclf import run_experiment

plan, preds, summary = run_experiment(
    dataset_path="Dataset/HumAID/california_wildfires_2018/california_wildfires_2018_train.tsv",
    rules=RULES_BASELINE,     # supply your chosen rules text from rules/
    model="gpt-4o-mini",
    tag="modeS-RULES_BASELINE",
    dryrun_n=20,              # small synchronous sanity check
    poll_secs=60,             # batch status polling interval
    do_analysis=True,         # write analysis/ charts and metrics
)
summary
```

#### Resume a submitted run later

```python
from humaidclf import resume_experiment
plan2, preds2, summary2 = resume_experiment(plan["dir"])   # uses batch_meta.json in that directory
```

### Option 2 — Step-by-step

#### A) Load data (TSV)

Expected columns: `tweet_id`, `tweet_text`, and (optional) `class_label` for ground truth.

```python
from dotenv import load_dotenv; load_dotenv()
from humaidclf import load_tsv

dataset_path = "Dataset/HumAID/california_wildfires_2018/california_wildfires_2018_train.tsv"
df = load_tsv(dataset_path, id_col="tweet_id", text_col="tweet_text", label_col="class_label")
len(df), df.head(2)
```

#### B) Choose rules (zero-shot)

Rules live **outside** the package in the `rules/` module so you can iterate freely.

```python
from rules import RULES_1  # or RULES_BASELINE
RULES = RULES_1
```

#### C) Dry-run a small sample (sanity check)

```python
from humaidclf import sync_test_sample, macro_f1
demo = sync_test_sample(df, n=20, rules=RULES, model="gpt-4o-mini", temperature=0.0, seed=42)
print("Sample Macro-F1:", macro_f1(demo))
demo.head()
```

#### D) Plan output directories & build the Batch requests

```python
from humaidclf import plan_run_dirs, build_requests_jsonl_S

plan = plan_run_dirs(dataset_path, out_root="runs", model="gpt-4o-mini", tag="modeS-RULES1")
build_requests_jsonl_S(df, plan["requests_jsonl"], rules=RULES, model=plan["model"], temperature=0.0)
plan
```

This creates:

```
runs/<event>/<split>/gpt-4o-mini/<timestamp>-modeS-RULES1/
  requests.jsonl
  (later) outputs.jsonl, predictions.csv, analysis/
```

#### E) Submit Batch, wait, download, parse

```python
import json

from humaidclf import (
    upload_file_for_batch,
    create_batch,
    wait_for_batch,
    download_file_content,
    parse_outputs_S_to_df,
    macro_f1,
)

# Submit batch and persist IDs so you can resume later
fid = upload_file_for_batch(str(plan["requests_jsonl"]))
bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")

with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
    json.dump({"file_id": fid, "batch_id": bid}, f, indent=2)

# Poll until the batch is done (completed/failed/cancelled)
info = wait_for_batch(bid, poll_secs=60)

# Guard: only proceed if completed
status = info.get("status")
if status != "completed":
    raise RuntimeError(f"Batch ended with status='{status}'. Full info:\n{json.dumps(info, indent=2)}")

# Download outputs and parse to predictions.csv
out_file_id = info["output_file_id"]
download_file_content(out_file_id, str(plan["outputs_jsonl"]))

preds = parse_outputs_S_to_df(plan["outputs_jsonl"], df)
preds.to_csv(plan["predictions_csv"], index=False)
print("Saved predictions to:", plan["predictions_csv"])
print("Macro-F1:", macro_f1(preds))
```

---

## Token budgeting & dataset gating (`budget.py`)

Before you submit a Batch job, estimate tokens per dataset and decide which ones fit your tier cap.

```python
from pathlib import Path
import pandas as pd
from rules import RULES_1
from humaidclf import build_token_index  # from budget.py

BASE = Path("Dataset/HumAID")
SPLITS = ["train"]                # or ["train","dev","test"]
MODEL = "gpt-4o-mini"
BATCH_TOKEN_LIMIT = 2_000_000     # e.g., Tier-1 cap
SAFETY_MARGIN = 0.90              # 10% headroom

def discover_tsvs(base: Path, splits: list[str]):
    items = []
    for event_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        event = event_dir.name
        for split in splits:
            tsv = event_dir / f"{event}_{split}.tsv"
            if tsv.exists():
                items.append({"event": event, "split": split, "tsv": str(tsv)})
    return pd.DataFrame(items)

df_sources = discover_tsvs(BASE, SPLITS)
token_index = build_token_index(
    df_sources, model=MODEL, rules_text=RULES_1,
    batch_token_limit=BATCH_TOKEN_LIMIT, safety_margin=SAFETY_MARGIN,
    sample_size=200, max_output_tokens=40
)
display(token_index)

df_fit     = token_index[token_index["fits_cap"]]
df_too_big = token_index[~token_index["fits_cap"]]
```

Optionally shard a large TSV into chunks that each fit your budget:

```python
from humaidclf import shard_dataset_by_tokens

TARGET_BUDGET = int(BATCH_TOKEN_LIMIT * SAFETY_MARGIN)
shards = shard_dataset_by_tokens(
    "Dataset/HumAID/some_event/some_event_train.tsv",
    model=MODEL, rules_text=RULES_1, target_token_budget=TARGET_BUDGET, max_output_tokens=40
)
len(shards), [len(s) for s in shards]
```

---

## Switching API keys (`batch.py`)

You can switch keys globally or just for a code block. This lets you run small datasets with your Tier‑1 key and large ones with your alternate key.

```python
from humaidclf.batch import (
    set_api_key_env, set_api_key_value, get_active_api_key_label,
    use_api_key_env, use_api_key_value,
)

# Permanent switch for the session (using .env variable names)
set_api_key_env("OPENAI_API_KEY_1")
print(get_active_api_key_label())  # -> "OPENAI_API_KEY_1"

# Temporary switch (auto-restores after the 'with' block)
with use_api_key_env("OPENAI_API_KEY_2"):
    print(get_active_api_key_label())  # -> "OPENAI_API_KEY_2"
    # run big batches here
print(get_active_api_key_label())      # restored
```

Combine with token budgeting to route datasets to the right key:

```python
from humaidclf import run_experiment
from rules import RULES_1

# 1) Run "fit" datasets on Tier-1 key
with use_api_key_env("OPENAI_API_KEY_1"):
    for _, row in df_fit.iterrows():
        run_experiment(row["tsv"], rules=RULES_1, model="gpt-4o-mini", tag="modeS-RULES1-TIER1")

# 2) Run "too big" datasets on alternate key
with use_api_key_env("OPENAI_API_KEY_2"):
    for _, row in df_too_big.iterrows():
        run_experiment(row["tsv"], rules=RULES_1, model="gpt-4o-mini", tag="modeS-RULES1-ALT")
```

---

## Curated results & summary page (`report.py`) — NEW

You can **manually curate** the best runs under `results/` and build a dashboard page.

**Directory layout (curated):**
```
results/<event>/<split>/<model>/<run_name>/
  predictions.csv
  analysis/
    charts/
      confusion_matrix_counts.png
      confusion_matrix_row_normalized.png
      per_class_error_rate.png
      per_class_f1.png
      top_confusions.png        # optional
      confusion_matrix.csv
      confusion_matrix_row_normalized.csv
      per_class_metrics.csv
      summary.json
    mistakes.csv
```

**Build / refresh the page (`results/index.html`)**
```python
from humaidclf.report import build_results_index
df_summary = build_results_index("results", out_html="results/index.html")
df_summary.head()
```

**Optional helper to copy a run you like into `results/`:**
```python
from humaidclf.report import promote_run_to_results

promote_run_to_results(
    run_dir="runs/<event>/<split>/<model>/<timestamp>-<tag>",
    results_root="results",
    run_name="<timestamp>-<tag>"   # or any label you prefer
)
```
The generated HTML includes:
- A **summary table** (Event, Split, Model, Run, Test size, Accuracy, Macro‑F1)
- Detailed cards per result with **zoomable charts** (click any image to open a modal, ESC to close)

**Notebook (convenience):** `00_build_results_index.ipynb`  
Minimal contents:
```python
from humaidclf.report import build_results_index
df = build_results_index("results", out_html="results/index.html")
df
```
(Optional) add a second cell to preview inline (but images cannot be displayed):
```python
from IPython.display import HTML
HTML(filename="results/index.html")
```

---

## Analysis

### A) Analyze the current run

```python
from humaidclf import analyze_and_export_mistakes

mistakes_df, summary, per_cls, conf_df = analyze_and_export_mistakes(
    pred_csv_path=plan["predictions_csv"],
    out_mistakes_csv_path=plan["dir"] / "analysis" / "mistakes.csv",
    charts_dir=plan["dir"] / "analysis" / "charts",
)
summary
```

Artifacts appear under `runs/.../<run_id>/analysis/` as listed in the layout above.

### B) Analyze a past run from a path

```python
from pathlib import Path
from humaidclf import analyze_and_export_mistakes

base = Path("runs/california_wildfires_2018/train/gpt-4o-mini/20251017-220548-modeS-RULES1")
mistakes_df, summary, per_cls, conf_df = analyze_and_export_mistakes(
    pred_csv_path=base / "predictions.csv",
    out_mistakes_csv_path=base / "analysis" / "mistakes.csv",
    charts_dir=base / "analysis" / "charts",
)
summary
```

---

## Exported API (from `humaidclf/__init__.py`)

```python
# IO
load_tsv, plan_run_dirs

# Prompts
LABELS, SYSTEM_PROMPT, make_user_message

# Batch
SCHEMA_S, sync_test_sample, build_requests_jsonl_S,
upload_file_for_batch, create_batch, get_batch, wait_for_batch,
download_file_content, parse_outputs_S_to_df,
# API key switching
set_api_key_env, set_api_key_value, get_active_api_key_label,
use_api_key_env, use_api_key_value,

# Budgeting
get_token_encoder, estimate_request_tokens, estimate_dataset_tokens,
build_token_index, shard_dataset_by_tokens,

# Eval
macro_f1, analyze_and_export_mistakes

# Report
promote_run_to_results, build_results_index

# Runners
run_experiment, resume_experiment
```

---

## Cost & Quality Tips

- Rules length vs cost: shorter rules reduce prompt tokens; keep only what moves the metric.
- `max_tokens`: 20–40 is enough for a single JSON label.
- Temperature: use 0.0 for deterministic classification.
- Macro-F1 vs Accuracy: Macro-F1 treats all classes equally; accuracy can be inflated by majority classes.
- Confusion matrices:
  - Counts: shows frequencies (can be dominated by big classes).
  - Row-normalized: shows per-class recall (diagonals = recall), fixed [0,1] with colorbar.
- Large datasets: use `build_token_index()` to check caps, `use_api_key_env()` to switch keys, and `shard_dataset_by_tokens()` if you want to keep everything under one key.

---

## Security & Git Tips

- Never commit `.env` or keys. If you did, rotate the key and remove it from history.
- Long commit messages: `git commit -m "title" --edit` (Vim: paste -> Esc :wq) or `git commit -F COMMIT_MESSAGE.txt`.

---

## License
Code is released under the MIT License (see `LICENSE`).

**Data:** This repository does not relicense any datasets. Use of HumAID or other datasets must comply with their original licenses/terms.  
**Services:** Use of the OpenAI API must comply with OpenAI's terms.

---

## Citation

If you use this code, please cite our forthcoming paper (details TBA).  
Provisional repository citation:

```bibtex
@misc{humaid-zeroshot-2025,
  title        = {Zero-shot HumAID Tweet Classification with OpenAI Batch},
  author       = {Anh Tran and Hongmin Li},
  year         = {2025},
  howpublished = {\url{https://github.com/anhtranst/humaid}},
  note         = {Version 0.1.0, MIT License}
}
```
